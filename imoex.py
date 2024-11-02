import requests

# URL для получения данных по акциям и облигациям
stocks_url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json'
bonds_url = 'https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQOB/securities.json'

def get_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Ошибка: статус-код {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

# Функция для извлечения параметров акций
def extract_stock_data(data):
    stocks = data.get('securities', {}).get('data', [])
    stock_info = []
    for stock in stocks:
        info = {
            'name': stock[2],  # Название акции
            'average_trading_volume': stock[12] if len(stock) > 12 else "Нет данных",  # Средний дневной объем торгов
            'dividend_yield': stock[15] if len(stock) > 15 else "Нет данных",  # Дивидендная доходность
            'dividend_history': stock[16] if len(stock) > 16 else "Нет данных",  # История дивидендов
            'year_high': stock[18] if len(stock) > 18 else "Нет данных",  # Годовой максимум
            'year_low': stock[19] if len(stock) > 19 else "Нет данных",  # Годовой минимум
            'sector': stock[5] if len(stock) > 5 else "Нет данных",  # Сектор
            'index_status': stock[6] if len(stock) > 6 else "Нет данных",  # Индексный статус (включение в индексы)
            'market_cap': stock[9] if len(stock) > 9 else "Нет данных",  # Рыночная капитализация
        }
        stock_info.append(info)
    return stock_info

# Функция для извлечения кредитного рейтинга и доходности для погашения облигаций
def extract_bond_data(data):
    bonds = data.get('securities', {}).get('data', [])
    bond_info = []
    for bond in bonds:
        info = {
            'name': bond[2],  # Название облигации
            'credit_rating': bond[15] if len(bond) > 15 else "Нет данных",  # Кредитный рейтинг
            'yield_to_maturity': bond[9] if len(bond) > 9 else "Нет данных"  # Доходность к погашению
        }
        bond_info.append(info)
    return bond_info

# Получаем данные по акциям и облигациям
stocks_data = get_data(stocks_url)
bonds_data = get_data(bonds_url)

# Извлекаем и выводим требуемую информацию отдельно
if stocks_data:
    print("Акции (средний объём торгов, дивиденды, ценовой диапазон, сектор, индексный статус, капитализация):")
    stocks_info = extract_stock_data(stocks_data)
    for stock in stocks_info:
        print(stock)

if bonds_data:
    print("\nОблигации (кредитный рейтинг и доходность для погашения):")
    bonds_info = extract_bond_data(bonds_data)
    for bond in bonds_info:
        print(bond)