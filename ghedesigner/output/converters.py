from math import floor

from ghedesigner.constants import HRS_IN_DAY


def hours_to_month(hours: float) -> float:
    """
    Convert a number of hours into fractional months, assuming a non-leap year calendar.
    """
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hrs_year = [HRS_IN_DAY * d for d in days]
    total_year = sum(hrs_year)
    years = floor(hours / total_year)
    rem = hours - years * total_year

    # find month index
    total = 0
    month = 0
    for idx, h in enumerate(hrs_year):
        total += h
        if rem <= total:
            month = idx
            break

    prev_total = total - hrs_year[month]
    month_frac = (rem - prev_total) / hrs_year[month]
    return years * 12 + month + month_frac


def ghe_time_convert(hours: int) -> tuple[int, int, int]:
    """
    Convert total hours into (month, day, hour) tuple in a non-leap year.
    """
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hrs_year = [HRS_IN_DAY * d for d in days]

    total = 0
    month = 0
    for idx, h in enumerate(hrs_year):
        if hours < total + h:
            month = idx
            break
        total += h

    hr_into_month = hours - total
    day = floor(hr_into_month / HRS_IN_DAY) + 1
    hour_in_day = (hr_into_month % HRS_IN_DAY) + 1
    return month + 1, day, int(hour_in_day)
