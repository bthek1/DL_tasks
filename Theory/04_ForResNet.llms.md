# Idea - ForResNet (coming soon)

ForResNet

Author

Benedict Thekkel

``` python
from zoneinfo import ZoneInfo  # Built-in from Python 3.9+
from datetime import datetime as dt, UTC
```

``` python
# 11PM Brisbane Time converted to UTC
brisbane_tz = ZoneInfo("Australia/Brisbane")
daily_time_brisbane = dt.now(brisbane_tz).replace(hour=23, minute=0, second=0, microsecond=0)

# Convert to UTC
daily_time_utc = daily_time_brisbane.astimezone(UTC)
```

``` python
brisbane_tz
```

    zoneinfo.ZoneInfo(key='Australia/Brisbane')

``` python
import pytz
```

Back to top
