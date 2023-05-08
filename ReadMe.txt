Inside your terminal and can't remember who Arsenal play next or what channel it's on? Perfect. 

Installation
Once cloned hop into the repo and run:
virtualenv .env && source .env/bin/activate && pip install -r requirements.txt

Then run:
python3 ars.py

Bonus: Add the script as an alias 'arsenal'

Workings:
web crawls https://www.live-footballontv.com/arsenal-on-tv.html and saves you a job
