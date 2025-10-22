import holidays
from datetime import date, datetime

class HungarianWorkdayAnalyzer:
    """
    Analyze Hungarian holidays and working days using the holidays library.
    Handles both official holidays and working day adjustments.
    """
    
    def __init__(self):
        # Initialize Hungarian holidays cache for different years
        self._holidays_cache = {}
        self._extra_working_holidays = [     
            date(2024, 3, 29),  # Easter Friday 2024
            date(2024, 4, 1),  # Easter Monday 2024
            date(2024, 5, 20),  # Whit Monday 2024
            date(2024, 8, 19),  # extra day for August 20 holiday 2024
            date(2024, 12, 24),  # Christmas Eve 2024
            date(2024, 12, 27),  # extra day for Christmas 2024
            date(2024, 11, 1),  # All Saints' Day 2024
            date(2025, 4, 18),  # Easter Friday 2025
            date(2025, 5, 2),  # extra day for Labour Day 2025
            date(2025, 4, 21),  # Easter Monday 2025
            date(2025, 6, 9),  # Whit Monday 2025
            date(2025, 10, 24),  # extra day for October 23 holiday 2025
            date(2025, 12, 24),  # Christmas Eve 2025
        ]
        self.extra_working_days = [
            date(2024, 8, 3), #instead of Aug 19
            date(2024, 12, 7), #instead of Dec 24
            date(2024, 12, 14), #instead of Dec 27
            date(2025, 5, 17), #instead of May 2
            date(2025, 10, 18), #instead of Oct 23
            date(2025, 12, 13), #instead of Dec 24
        ]
    
    def _get_holidays_for_year(self, year):
        """Get Hungarian holidays for a specific year, using cache."""
        if year not in self._holidays_cache:
            self._holidays_cache[year] = holidays.Hungary(years=year)
        return self._holidays_cache[year]
    
    def is_official_holiday(self, date_obj):
        """
        Check if a date is an official Hungarian holiday or extra working holiday.
        
        Args:
            date_obj: datetime.date, pandas.Timestamp, or datetime.datetime object
            
        Returns:
            bool: True if the date is an official holiday or extra working holiday
        """
        # Convert to date object if needed
        if hasattr(date_obj, 'date') and callable(date_obj.date):
            date_obj = date_obj.date()
        elif hasattr(date_obj, 'to_pydatetime'):
            date_obj = date_obj.to_pydatetime().date()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        # Check official holidays from holidays library
        hun_holidays = self._get_holidays_for_year(date_obj.year)
        if date_obj in hun_holidays:
            return True
        
        # Check extra working holidays (bridge days, etc.)
        if date_obj in self._extra_working_holidays:
            return True
        
        return False
    
    def is_weekend(self, date_obj):
        """Check if date falls on weekend (Saturday or Sunday)."""
        # Convert to date object if needed
        if hasattr(date_obj, 'date') and callable(date_obj.date):
            date_obj = date_obj.date()
        elif hasattr(date_obj, 'to_pydatetime'):
            date_obj = date_obj.to_pydatetime().date()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_working_day(self, date_obj):
        """
        Determine if a date is a working day in Hungary.
        
        Takes into account:
        - Official holidays and extra working holidays
        - Weekends
        - Extra working days (compensatory work days)
        
        Args:
            date_obj: datetime.date, pandas.Timestamp, or datetime.datetime object
            
        Returns:
            bool: True if it's a working day, False if it's a holiday/non-working day
        """
        # Convert to date object if needed
        if hasattr(date_obj, 'date') and callable(date_obj.date):
            date_obj = date_obj.date()
        elif hasattr(date_obj, 'to_pydatetime'):
            date_obj = date_obj.to_pydatetime().date()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        # Check if it's a designated extra working day (overrides holidays and weekends)
        if date_obj in self.extra_working_days:
            return True
        
        # Check if it's an official holiday or extra working holiday
        if self.is_official_holiday(date_obj):
            return False
        
        # Check if it's a weekend
        if self.is_weekend(date_obj):
            return False
        
        # If neither holiday nor weekend, it's a working day
        return True
    
    def get_holiday_name(self, date_obj):
        """
        Get the name of the holiday for a given date.
        
        Returns:
            str or None: Holiday name if it's a holiday, None otherwise
        """
        # Convert to date object if needed
        if hasattr(date_obj, 'date') and callable(date_obj.date):
            date_obj = date_obj.date()
        elif hasattr(date_obj, 'to_pydatetime'):
            date_obj = date_obj.to_pydatetime().date()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        hun_holidays = self._get_holidays_for_year(date_obj.year)
        return hun_holidays.get(date_obj)
