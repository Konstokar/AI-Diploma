import torch
import torch.nn as nn
import torch.optim as optim
import requests
import json

stocks_url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json'
bonds_url = 'https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQOB/securities.json'


def get_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка: статус-код {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None


# Функция для получения текущей цены акции по тикеру
def get_stock_price_per_share(ticker: str):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            last_price = data['marketdata']['data'][0][12]  # Цена последней сделки
            return last_price
        except IndexError:
            return "Цена не найдена или тикер некорректен."
    else:
        return "Ошибка при запросе к API."


# Функция для извлечения параметров акций
def extract_stock_data(data):
    stocks = data.get('securities', {}).get('data', [])
    stock_info = []
    for stock in stocks:
        ticker = stock[0] if len(stock) > 0 else "Нет данных"
        info = {
            'name': stock[2],
            'ticker': ticker,
            'dividend': {
                'yield': stock[15] if len(stock) > 15 else "Нет данных",
                'frequency': stock[16] if len(stock) > 16 else "Нет данных"
            }
        }
        stock_info.append(info)
    return stock_info


# Функция для извлечения параметров облигаций
def extract_bond_data(data):
    bonds = data.get('securities', {}).get('data', [])
    bond_info = []
    for bond in bonds:
        info = {
            'name': bond[2],
            'price': bond[10] if len(bond) > 10 and isinstance(bond[10], (int, float)) else "Нет данных",
            'coupon': {
                'size': bond[15] if len(bond) > 15 else "Нет данных",
                'frequency': bond[16] if len(bond) > 16 else "Нет данных"
            }
        }
        bond_info.append(info)
    return bond_info


# Загружаем данные с API
stocks_data = get_data(stocks_url)
bonds_data = get_data(bonds_url)

stock_info = extract_stock_data(stocks_data)
bond_info = extract_bond_data(bonds_data)


# Определение модели нейросети
class RiskClassifier(nn.Module):
    def __init__(self):
        super(RiskClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Подготовка данных для обучения
def prepare_data(stock_data, bond_data):
    X = []
    y = []

    for stock in stock_data[:10]:
        try:
            dividend_yield = float(stock['dividend']['yield'])
        except (ValueError, TypeError):
            dividend_yield = 0

        X.append([0, dividend_yield])
        y.append(0)

    for bond in bond_data[:10]:
        try:
            ytm = float(bond['price'])
        except (ValueError, TypeError):
            ytm = 0
        credit_rating = bond['coupon']['frequency']
        if credit_rating == "AAA":
            credit_rating = 1
        elif credit_rating == "AA":
            credit_rating = 2
        else:
            credit_rating = 3

        X.append([ytm, credit_rating])
        y.append(1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Создаем данные
X_train, y_train = prepare_data(stock_info, bond_info)

# Создаем и обучаем модель
model = RiskClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    print("Модель обучена")


train_model(model, criterion, optimizer, X_train, y_train)
torch.save(model.state_dict(), 'risk_classifier_model.pth')
print("Модель сохранена в risk_classifier_model.pth")


# Функция для прогнозирования и сохранения результатов в JSON
def predict_and_save_results(model, stock_data, bond_data):
    model.eval()
    stock_predictions = []
    bond_predictions = []

    # Прогноз для акций
    for stock in stock_data[:10]:  # Берем только 10 акций
        try:
            dividend_yield = float(stock['dividend']['yield'])
        except (ValueError, TypeError):
            dividend_yield = 0

        input_data = torch.tensor([[0, dividend_yield]], dtype=torch.float32)
        output = model(input_data)
        risk_level = torch.argmax(output, dim=1).item()
        stock_predictions.append((stock, risk_level))

    # Прогноз для облигаций
    for bond in bond_data[:10]:  # Берем только 10 облигаций
        try:
            price = float(bond['price'])
        except (ValueError, TypeError):
            price = 0

        coupon_freq = bond['coupon']['frequency']
        if coupon_freq == "AAA":
            coupon_freq = 1
        elif coupon_freq == "AA":
            coupon_freq = 2
        else:
            coupon_freq = 3

        input_data = torch.tensor([[price, coupon_freq]], dtype=torch.float32)
        output = model(input_data)
        risk_level = torch.argmax(output, dim=1).item()
        bond_predictions.append((bond, risk_level))

    # Создание структуры JSON
    result = {
        "Stocks": {"Low": [], "Medium": [], "High": []},
        "Bonds": {"Low": [], "Medium": [], "High": []}
    }

    risk_map = {0: "Low", 1: "Medium", 2: "High"}

    for stock, risk_level in stock_predictions:
        # Получение цены только на этапе формирования JSON
        stock['price'] = get_stock_price_per_share(stock['ticker'])
        result["Stocks"][risk_map[risk_level]].append(stock)

    for bond, risk_level in bond_predictions:
        result["Bonds"][risk_map[risk_level]].append(bond)

    # Сохранение результата в JSON-файл
    with open('risk_assessment.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("Результаты сохранены в risk_assessment.json")


# Выполнение функции сохранения
predict_and_save_results(model, stock_info, bond_info)