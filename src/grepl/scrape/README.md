# Using the scraper now (headless = False)

Just
```
uv run python src/grepl/scrape/scrape_outlier.py
```

And log into the window. Navigate the main outlierdb page (click x then navigate home)

and press enter to let her rip

This seems to be better than the way below:
* don't have to quit other chrome windows (afaict)
* don't have to run that 9223 command

Expecting this to blow up in my face tonight but so far so good...
-----

# Using the scraper

Quit all Chrome windows, then run this in the terminal:
```
# first quit all Chrome windows
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9223 \
  --user-data-dir="$HOME/chrome-debug-profile"
```
This should open a new chrome window.

Then run the scraper:

```
uv run python src/grepl/scrape/scrape_outlier.py
```

You'll be taken to the outlierdb.com login page. You probably don't even have to actually enter your login info - there should be a red "X" that you can click on, and you'll be taken to the outlierdb.com front page. Press enter to continue.

We expect 20 videos per page. The scraper will scroll down the page 20 times, and save the page source to a sqlite database - `outlierdb.sqlite`.

To see what we've got, run:

```
sqlite3 outlierdb.sqlite
```

And then:

```
select count(distinct content) from raw_page;
```

Should print 20 (for as many pages as you scraped for).

To parse the pages, run:

```
uv run python src/grepl/scrape/parse_outlier.py
```

And then to parse those into video URLs, run:

```
uv run python src/grepl/scrape/parse_video_url.py
```

And then to download the video clips, run (for example)

```
uv run python src/grepl/scrape/video_clip_downloader.py https://www.youtube.com/watch\?v\=WjjnS5MxJws\&t\=56s
```

This will download a 60s clip starting at 56 seconds into the video.

The clips will be saved to `clips/`. A clip is on avg ~2 MB.
