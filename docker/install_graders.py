"""Build-time only: official packages from source-registry commits."""
import json,subprocess,sys
from pathlib import Path
s=json.loads(Path('/app/configs/ordinary/sources.json').read_text())['sources']
def pip(*args):subprocess.run([sys.executable,'-m','pip',*args],check=True)
for key in ('evalplus_grader','ifbench_grader'):
    spec=s[key]
    pip('install','git+'+spec['url']+'@'+spec['revision'])
pip('install','math-verify==0.9.0','absl-py','langdetect','immutabledict','nltk','emoji',
    'syllapy','pandas','scipy','numpy','six','tqdm','pythainlp','datasets==5.2.0')
import nltk
for name in ('punkt','punkt_tab','wordnet','omw-1.4','stopwords','averaged_perceptron_tagger_eng'):
    if not nltk.download(name,download_dir='/opt/nltk_data',raise_on_error=True):
        raise RuntimeError('NLTK resource unavailable: '+name)
# IFBench creates .nltk_data beside its module at import; create it in the image.
nltk.data.path.insert(0,'/opt/nltk_data')
import ifbench.instructions
# Multi-IF imports Thai tokenizers even in its English subset.
import os
os.environ['PYTHAINLP_DATA']='/opt/pythainlp_data'
import pythainlp.tokenize
pip('check')
