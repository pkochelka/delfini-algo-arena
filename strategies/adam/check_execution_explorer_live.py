"""Check the updated explorer's three real Round 1 callbacks without browser automation."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8062')
url = parser.parse_args().url.rstrip('/')
for tab in ('queue', 'maker', 'latency'):
    inputs = [('tab', 'value', tab), ('tape', 'value', 'round1'), ('window', 'data', None),
              ('dt', 'value', 1.0), ('exec-side', 'value', 'both'), ('exec-size', 'value', 'all'),
              ('exec-placement', 'value', 'touch'), ('exec-horizon', 'value', 1.0),
              ('exec-latency', 'value', 5.0), ('exec-cancel-age', 'value', 100.0)]
    payload = {'output': 'content.children', 'outputs': {'id': 'content', 'property': 'children'},
               'inputs': [{'id': i, 'property': p, 'value': v} for i, p, v in inputs],
               'state': [], 'changedPropIds': ['tab.value']}
    request = Request(url + '/_dash-update-component', json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
        assert 'content' in result['response']
        print(f'{tab}: HTTP {response.status}, content rendered for Round 1', flush=True)
