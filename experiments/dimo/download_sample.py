"""Download the official CC BY 4.0 DIMO sample and CAD; preserve attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import requests

FILES = {
    'dimo.tar': 'https://datasets.ilabt.imec.be/dimo/tar/dimo.tar',
    'dimo_sample.tar': 'https://datasets.ilabt.imec.be/dimo/tar/dimo_sample.tar',
}

def download(url, path):
    if path.exists():
        print('Using existing '+path.name,flush=True)
        return
    partial=path.with_suffix(path.suffix+'.partial')
    with requests.get(url,stream=True,timeout=(20,60)) as response:
        response.raise_for_status()
        size=int(response.headers.get('Content-Length',0))
        if size>450_000_000:
            raise RuntimeError('Unexpected archive size; refusing bulk dataset download')
        total=0;last=time.monotonic()
        with partial.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                output.write(chunk);total+=len(chunk)
                if time.monotonic()-last>8:
                    print(f'{path.name}: {total/1e6:.1f}/{size/1e6:.1f} MB',flush=True)
                    last=time.monotonic()
        if size and total!=size:
            raise RuntimeError('Incomplete download')
    partial.replace(path)
    print(f'Downloaded {path.name}: {total/1e6:.1f} MB',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--full',action='store_true',help='Download the entire 368 MB sample; default streams only its first L515 scene')
    args=parser.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in FILES.items():
        if name=='dimo_sample.tar' and not args.full:
            destination=args.cache/'selected'
            destination.mkdir(exist_ok=True)
            partial=args.cache/'dimo_sample.tar.partial'
            prefix='dimo_sample/real_l515/'
            selected=[];seen=False
            if partial.exists():
                source=partial.open('rb'); response=None
            else:
                response=requests.get(url,stream=True,timeout=(20,60));response.raise_for_status()
                response.raw.decode_content=True;source=response.raw
            try:
                with tarfile.open(fileobj=source,mode='r|') as archive:
                    for member in archive:
                        if member.name.startswith(prefix):
                            seen=True
                            if member.isfile() and not Path(member.name).name.startswith('._'):
                                archive.extract(member,destination,filter='data')
                                p=destination/member.name
                                selected.append({'name':member.name,'bytes':member.size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
                        elif seen:break
            finally:
                source.close()
                if response:response.close()
            if len([m for m in selected if '/rgb/' in m['name']])!=13 or len([m for m in selected if '/depth/' in m['name']])!=13:
                raise RuntimeError('Expected thirteen complete L515 RGB/depth pairs')
            records.append({'url':url,'selection':'first real_l515 scene, 13 RGB/depth pairs','members':selected,'extracted_to':str(destination.resolve())})
            print(f'Extracted {len(selected)} selected files; skipped remaining cameras/synthetic archive',flush=True)
            continue
        path=args.cache/name;download(url,path)
        destination=args.cache/name.replace('.tar','_unpacked')
        destination.mkdir(exist_ok=True)
        with tarfile.open(path,'r:') as archive:
            members=archive.getmembers()
            if sum(m.size for m in members)>2_000_000_000:
                raise RuntimeError('Unexpected extracted size')
            archive.extractall(destination,filter='data')
            print(f'Extracted {len(members)} archive members to {destination}',flush=True)
        records.append({'url':url,'bytes':path.stat().st_size,
                        'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
                        'extracted_to':str(destination.resolve())})
    provenance={'dataset':'Dataset of Industrial Metal Objects (DIMO)',
                'authors':['Peter De Roovere','Steven Moonen','Nick Michiels','Francis Wyffels'],
                'project':'https://pderoovere.github.io/dimo/',
                'license':'CC BY 4.0','license_source':'https://pderoovere.github.io/dimo/usage/',
                'license_url':'https://creativecommons.org/licenses/by/4.0/',
                'paper':'https://arxiv.org/abs/2208.04052','files':records}
    (args.cache/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
