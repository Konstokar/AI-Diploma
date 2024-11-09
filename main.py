import torch
import torch.nn as nn
import torch.optim as optim
import requests
import json
import concurrent.futures
from datetime import datetime

# URL для загрузки данных акций и облигаций
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


# Функция для получения цен акций или облигаций по тикерам с многопоточностью
def get_prices(tickers, is_stock=True):
    base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/" if is_stock else "https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQOB/securities/"
    prices = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(requests.get, base_url + f"{ticker}.json"): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                response = future.result()
                if response.status_code == 200:
                    data = response.json()
                    last_price = data['marketdata']['data'][0][12]
                    trade_date = data['marketdata']['data'][0][6]
                    current_date = datetime.now().strftime('%Y-%m-%d')

                    # Если текущих торгов нет, использовать цену последних торгов
                    if trade_date == current_date:
                        prices[ticker] = last_price
                    else:
                        prices[ticker] = f"Последняя доступная цена: {last_price} от {trade_date}"
                else:
                    prices[ticker] = "Цена не найдена"
            except Exception as e:
                prices[ticker] = f"Ошибка при запросе: {e}"
    return prices


# Функция для вычисления периодичности выплат
def calculate_coupon_frequency_per_year(frequency: str):
    if frequency == "Раз в полгода":
        return 2
    elif frequency == "Ежеквартально":
        return 4
    elif frequency == "Ежемесячно":
        return 12
    else:
        return 1  # По умолчанию ежегодная выплата


# Функция для извлечения данных акций
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


# Функция для извлечения данных облигаций
def extract_bond_data(data):
    bonds = data.get('securities', {}).get('data', [])
    bond_info = []
    for bond in bonds:
        ticker = bond[0] if len(bond) > 0 else "Нет данных"
        coupon_frequency = bond[16] if len(bond) > 16 else "Нет данных"
        info = {
            'name': bond[2],
            'ticker': ticker,
            'coupon': {
                'size': bond[15] if len(bond) > 15 else "Нет данных",
                'frequency_per_year': calculate_coupon_frequency_per_year(coupon_frequency)
            },
            'maturity_date': bond[13] if len(bond) > 13 else "Нет данных"
        }
        bond_info.append(info)
    return bond_info


# Определение архитектуры нейросети для классификации риска
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
        coupon_freq = bond['coupon']['frequency_per_year']
        credit_rating = 1 if coupon_freq == "AAA" else 2 if coupon_freq == "AA" else 3
        X.append([0, credit_rating])
        y.append(1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Основной блок кода для выполнения
# Загрузка данных
stocks_data = get_data(stocks_url)
bonds_data = get_data(bonds_url)

stock_info = extract_stock_data(stocks_data)
bond_info = extract_bond_data(bonds_data)

# Создание и обучение модели
X_train, y_train = prepare_data(stock_info, bond_info)
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

# Функция для предсказания и сохранения результатов
def predict_and_save_results(model, stock_data, bond_data):
    model.eval()
    stock_predictions = []
    bond_predictions = []

    stock_tickers = [stock['ticker'] for stock in stock_data]
    bond_tickers = [bond['ticker'] for bond in bond_data]

    stock_prices = get_prices(stock_tickers, is_stock=True)
    bond_prices = get_prices(bond_tickers, is_stock=False)

    for stock in stock_data:
        try:
            dividend_yield = float(stock['dividend']['yield'])
        except (ValueError, TypeError):
            dividend_yield = 0
        input_data = torch.tensor([[0, dividend_yield]], dtype=torch.float32)
        output = model(input_data)
        risk_level = torch.argmax(output, dim=1).item()
        stock['price'] = stock_prices.get(stock['ticker'], "Нет данных")
        stock_predictions.append((stock, risk_level))

    for bond in bond_data:
        coupon_freq = bond['coupon']['frequency_per_year']
        credit_rating = 1 if coupon_freq == "AAA" else 2 if coupon_freq == "AA" else 3
        input_data = torch.tensor([[0, credit_rating]], dtype=torch.float32)
        output = model(input_data)
        risk_level = torch.argmax(output, dim=1).item()
        bond['price'] = bond_prices.get(bond['ticker'], "Нет данных")
        bond_predictions.append((bond, risk_level))

    result = {
        "Stocks": {"Low": [], "Medium": [], "High": []},
        "Bonds": {"Low": [], "Medium": [], "High": []}
    }

    risk_map = {0: "Low", 1: "Medium", 2: "High"}

    for stock, risk_level in stock_predictions:
        result["Stocks"][risk_map[risk_level]].append(stock)

    for bond, risk_level in bond_predictions:
        result["Bonds"][risk_map[risk_level]].append(bond)

    with open('risk_assessment.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("Результаты сохранены в risk_assessment.json")

# Выполнение функции предсказания и сохранения
predict_and_save_results(model, stock_info, bond_info)