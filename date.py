import requests


def get_last_trade_date(ticker: str) -> str:
    """
    Получает последнюю дату торговой сессии для заданного тикера с Московской Биржи.

    :param ticker: Тикер акции или облигации (например, "GAZP").
    :return: Дата последней торговой сессии в формате YYYY-MM-DD или сообщение об ошибке.
    """
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        data = response.json()

        # Ищем дату последней торговой сессии
        trade_date = data['marketdata']['data'][0][1]  # Поле с датой последней сессии (обычно в позиции [0][1])

        return trade_date
    except (requests.RequestException, IndexError, KeyError) as e:
        return f"Ошибка получения данных: {e}"


# Пример использования функции
ticker = "GAZP"  # Тикер для акции Газпрома
print(get_last_trade_date(ticker))