import requests
import json


def get_bond_info(ticker: str):
    url = f"https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities/{ticker}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Отладочный вывод, чтобы увидеть структуру данных
        print("Полученные данные:")
        print(json.dumps(data, indent=4, ensure_ascii=False))

        try:
            # Получаем колонку с названиями для marketdata и securities
            marketdata_columns = data['marketdata']['columns']
            securities_columns = data['securities']['columns']

            # Извлекаем значения, если они присутствуют
            marketdata = data['marketdata']['data'][0] if data['marketdata']['data'] else None
            securities = data['securities']['data'][0] if data['securities']['data'] else None

            if marketdata and securities:
                # Получаем цену последней сделки
                last_price = marketdata[
                    marketdata_columns.index('LAST')] if 'LAST' in marketdata_columns else "Нет данных"

                # Достаем информацию об облигации
                coupon_value = securities[
                    securities_columns.index('COUPONVALUE')] if 'COUPONVALUE' in securities_columns else "Нет данных"
                coupon_period = securities[
                    securities_columns.index('COUPONPERIOD')] if 'COUPONPERIOD' in securities_columns else "Нет данных"
                maturity_date = securities[
                    securities_columns.index('MATDATE')] if 'MATDATE' in securities_columns else "Нет данных"

                return {
                    "price": last_price,
                    "coupon_value": coupon_value,
                    "coupon_period": coupon_period,
                    "maturity_date": maturity_date
                }
            else:
                return "Данные по облигации отсутствуют."

        except (IndexError, ValueError):
            return "Информация по облигации не найдена или тикер некорректен."
    else:
        return "Ошибка при запросе к API."


ticker = "SU29009RMFS6"  # Пример тикера для облигации
bond_info = get_bond_info(ticker)

if isinstance(bond_info, dict):
    print(f"Цена облигации {ticker}: {bond_info['price']} руб.")
    print(f"Размер купона: {bond_info['coupon_value']} руб.")
    print(f"Частота выплат купона: каждые {bond_info['coupon_period']} дней")
    print(f"Срок погашения: {bond_info['maturity_date']}")
else:
    print(bond_info)