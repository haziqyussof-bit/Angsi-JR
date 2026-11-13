"""Local literature-grounded geoscience assistant when approved live model is unavailable.

Uses a curated ORIGINAL summary of public scientific references, NOT stored
copyrighted full papers. An optional user-initiated Crossref metadata request
sends ONLY a generic literature query (never proprietary dashboard records).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import numpy as np
import pandas as pd

LIBRARY=[
  dict(topic='4D seismic repeatability',tags='4d repeatability nrms acquisition noise qc cross equalization normalization',
       title='Improved normalization of time-lapse seismic data using NRMS repeatability',
       citation='Stephen (2014), Geophysical Prospecting',url='https://doi.org/10.1111/1365-2478.12109',
       principle='Repeatability outside a reservoir can be used to filter unreliable 4D response before history matching.'),
  dict(topic='4D seismic acquisition quality',tags='seismic registration 4d repeatability wavelet time phase amplitude',
       title='4D seismic co-processing and acquisition repeatability',
       citation='Journal of Petroleum Exploration and Production Technology (2022)',
       url='https://doi.org/10.1007/s13202-022-01541-x',
       principle='NRMS depends on acquisition and frequency content. Matching and QC are needed before geological interpretation.'),
  dict(topic='Time shifts',tags='time shift seismic phase difference pressure compaction velocity',
       title='Methods of measurement for 4D seismic post-stack time shifts',
       citation='Geophysical Prospecting (2020)',url='https://doi.org/10.1111/1365-2478.13022',
       principle='Time-shift estimates require appropriate windows and residual-moveout analysis; amplitude differences alone can mislead.'),
  dict(topic='4D reservoir processes',tags='4d reservoir saturation pressure production monitoring history rock physics',
       title='A framework for subsurface monitoring by integrating reservoir simulation with time-lapse seismic surveys',
       citation='Scientific Reports (2023)',url='https://doi.org/10.1038/s41598-023-40548-0',
       principle='Dynamic reservoir interpretations benefit from reservoir simulation and cross-disciplinary constraints.'),
  dict(topic='Uncertain 4D anomalies',tags='4d anomalies fluid saturation pressure uncertainty detectability sweep',
       title='Utilizing the onset of time-lapse changes: reservoir monitoring and characterization',
       citation='Geophysical Journal International (2014)',url='https://doi.org/10.1093/gji/ggt526',
       principle='Pressure and saturation effects are not uniquely resolved by time-lapse anomalies; consider alternative causes.'),
  dict(topic='Volumetric physics',tags='volumetric stoiip giip bg bo reserve thickness area oil gas porosity sw ntg',
       title='Petroleum in-place volume equations',
       citation='Wiley petroleum engineering book excerpt (2018)',
       url='https://catalogimages.wiley.com/images/db/pdf/9781119387947.excerpt.pdf',
       principle='In-place petroleum volume depends on closure/rock volume, net-to-gross, porosity, initial hydrocarbon saturation, and formation volume factor.'),
  dict(topic='Petroleum resources classification',tags='remaining reserves produced recoverable resources production history prms',
       title='Petroleum reserve and resource definitions (PRMS-aligned)',
       citation='South Australia Energy & Mining (2021)',
       url='https://www.energymining.sa.gov.au/industry/energy-resources/geology-and-prospectivity/petroleum-lexicon/reserve-and-resource-definitions',
       principle='Reserves require discovery, recoverability, commerciality and remaining volumes; preliminary in-place screens are not reserves.'),
  dict(topic='Gamma ray and lithology',tags='gamma ray GR API clean sand shale clay reservoir lithology volume of shale VSH GR log',
       title='Electrical and Radioactive Well Logging',
       citation='U.S. Geological Survey, Water-Supply Paper 1544-D',
       url='https://pubs.usgs.gov/wsp/1544d/report.pdf',
       principle='GR commonly increases in clay-rich shale and decreases in cleaner sands, but GR alone cannot prove reservoir or fluid phase.'),
  dict(topic='Petroleum resources management',tags='SPE PRMS resources reserves volumes cumulative production risk economic commercially recoverable',
       title='2018 PRMS Update Completed',
       citation='SPE Journal of Petroleum Technology (2018)',
       url='https://jpt.spe.org/2018-prms-update-completed',
       principle='Volumes in place, prospective resources and reserves are separate classifications; reserves require recoverability and commerciality.'),
  dict(topic='Petroleum reserve definition',tags='prms cumulative production recovery reserves classification resources',
       title='Classification and Methodology for Reserves Calculations',
       citation='U.S. Bureau of Ocean Energy Management',
       url='https://www.boem.gov/oil-gas-energy/resource-evaluation/classification-and-methodology-reserves-calculations',
       principle='Remaining recoverable must be tied to a defined development project and production history, not a static seismic-change map.'),
]

TOPIC_CHOICES={
  '4D seismic QC':'4D seismic repeatability NRMS cross equalization',
  'Rock physics and pressure':'4D seismic pressure saturation changes rock physics',
  'Seismic attributes':'time lapse seismic RMS amplitude horizon conformable attributes',
  'Facies prediction':'seismic amplitude bright spot sand shale impedance tuning',
  'Well ties':'synthetic seismogram seismic well tie checkshot sonic density',
  'Volumetrics':'in place gas oil volumetric formation volume factor net to gross',
  'Opportunity evaluation':'remaining recoverable oil reserves uncertainty reservoir 4D'}


def match_references(question, n=3):
    q=set(re.findall(r'[a-z0-9]+',str(question).lower()))-{'is','the','in','on','and','to','what','how','of','a','i','want','nak','aku','ni','apa','berapa','with','can'}
    scored=[]
    for r in LIBRARY:
        text=set(re.findall(r'[a-z0-9]+',(r['topic']+' '+r['tags']+' '+r['title']).lower()))
        score=len(q&text)
        if score:scored.append((score,r))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [r for _,r in scored[:n]] or [LIBRARY[0],LIBRARY[5]]


def _evidence_summary(evidence, selected='None'):
    if evidence is None:
        return 'No selected numerical dashboard record. No Angsi-specific observation is asserted.'
    if isinstance(evidence, pd.DataFrame):
        return f'{selected}: {len(evidence)} measured/derived rows available; inspect full table in dashboard.'
    if isinstance(evidence, list):
        txt=f'{selected}: {len(evidence)} selected records.'
        if evidence and isinstance(evidence[0],dict):
            out=[]
            for row in evidence[:3]:
                parts=[]
                for key in ('ID','Category','Evidence_score','Area_km2','STOIIP_MMstb','GIIP_MMscf','Inplace_MMboe','QC'):
                    v=row.get(key)
                    if v is not None and not isinstance(v,(dict,list)):
                        parts.append(f'{key}={v}')
                if parts:out.append(', '.join(parts))
            if out:txt+=' Examples: '+' | '.join(out)
        return txt
    if isinstance(evidence,dict):
        available=', '.join(map(str,list(evidence.keys())[:9]))
        return f'{selected}: selected record contains {available}. Those are derived descriptions, not validated predictions.'
    return f'{selected}: read-only record of type {type(evidence).__name__} present.'


def local_volume_evidence(runs):
    """Read local Streamlit-derived volumes only; NEVER call an external endpoint.

    Output is a compact anonymised-in-the-UI summary of locally calculated
    scenarios, not production or reservoir verification.
    """
    out=[]
    if not isinstance(runs,dict):return out
    for reservoir,run in runs.items():
        if not isinstance(run,dict):continue
        frame=run.get('volumes')
        if not isinstance(frame,pd.DataFrame) or frame.empty:continue
        phase=run.get('phase','Gas scenario')
        metric={'Gas scenario':'GIIP_MMscf','Oil scenario':'STOIIP_MMstb',
                'Model phase when valid':'Inplace_MMboe',
                'Unknown phase · alternatives':'Gas_only_MMscf'}.get(phase,'GIIP_MMscf')
        if metric not in frame:continue
        vals=pd.to_numeric(frame[metric],errors='coerce')
        if not vals.notna().any():continue
        out.append(dict(ID=str(reservoir),Category='CALCULATED IN-PLACE SCREENING SCENARIO',
            **{metric:round(float(vals.sum()),3)},
            QC='4D anomaly candidate areas; phase assumptions and closures NOT validated'))
    return out


def answer(question, evidence=None, selected='None', project=None):
    """Deterministic retrieval-informed explanations. Never impersonates an LLM."""
    q=str(question).strip()
    if not q:return 'Ask a question about 4D, QC, well logs, volumetrics, or the ranked opportunities.'
    if q.lower().strip('!?. ') in {'hi','hello','hey','hai','salam','assalamualaikum'}:
        return ('Hi! I can help with 4D seismic QC, reservoir facies, well ties, volumes, '
                'and the evidence behind a candidate. Choose a dashboard record if you want a field-specific check. '
                'Ask me, for example: **How can I distinguish a genuine 4D signal from survey noise?**')
    text=q.lower()
    if any(word in text for word in ('mmstb','mmscf','volume','volumetr','stoiip','giip','remain','recovery','reserve','banyak oil','berapa gas')):
        explanation=('**Volumetric workflow:** mapped candidate area × gross thickness gives gross rock volume. '
            'Partition by facies-dependent NTG, PHIE and (1−Sw); divide oil pore volume by Bo (MMstb) or gas pore volume '
            'by Bg (MMscf). **In-place is not remaining**. A remaining *recoverable scenario* needs an explicitly assumed '
            'recovery factor AND independently verified cumulative production. 25 MMscf of GIIP is not automatically '
            '25 MMscf still recoverable. The dashboard never converts 4D amplitude directly into produced volume.')
    elif any(word in text for word in ('sand','shale','facies','bright','dim','reservo ir','litholog')):
        explanation=('**Facies workflow:** 1995 baseline RMS P30/P70 split defines bright / intermediate / dim amplitude '
            '*hypotheses*, NOT sand/shale truth. Tuning, impedance contrast, gas effects, phase and acquisition also change amplitude. '
            'Use GR/VSHALE, density, sonic, PHIE and well tops to calibrate; keep well-centred bias out of survey-wide mapping.')
    elif any(word in text for word in ('gamma','gr log','resistiv','synthetic','well tie','sonic','density','rhob','nphi','acoust')):
        explanation=('**Well-log and synthetic-tie QC:** clean sands often show lower gamma ray than clay-rich shale, '
            'but mineralogy and radioactive sands can complicate the GR response. Use GR, resistivity, neutron-density, '
            'PHIE and Sw jointly. For a synthetic seismic tie, check sonic slowness units, density units, well datum, '
            'measured checkshot TWT, acoustic impedance, reflectivity, wavelet frequency/phase, seismic polarity '
            'and well deviation. A correlation coefficient alone cannot validate an event tie.')
    elif any(word in text for word in ('horizon','top','base','structure','fault','contour')):
        explanation=('**Horizon workflow:** check measured markers and checkshots; track the same phase-consistent reflector '
            'and inspect faults, pinch-outs and topology. A smoothed trend fill is MODEL_ONLY, not a validated horizon. '
            'Depth conversion using one checkshot is provisional. Compare high-confidence and exploratory picks separately.')
    elif any(word in text for word in ('4d','nrms','repeat','seismic','time shift','rms','production','anomal')):
        explanation=('**4D workflow:** co-register the 1995 baseline and 2006 monitor, inspect valid sample/mute masks, '
            'reference-window NRMS and local noise before interpreting difference maps. High difference does not prove depletion, '
            'and low difference does not prove unswept hydrocarbons. Pressure, saturation, compaction and processing are alternatives.')
    else:
        explanation=('**Investigation approach:** formulate a testable geological hypothesis, map a reproducible observation '
            'from the appropriate seismic, marker and log data, evaluate alternatives and uncertainty, then choose the next QC '
            'or reservoir-engineering check. A convincing candidate needs more than an attractive anomaly map.')
    refs=match_references(q)
    citations='\n'.join(f"- [{r['citation']}: {r['title']}]({r['url']}) — {r['principle']}" for r in refs)
    return (f'**Scientific explanation**\n\n{explanation}\n\n'
       f'**Selected dashboard evidence**\n\n{_evidence_summary(evidence,selected)}\n\n'
       '**Confidence and next action**\n\nGeneral technical principles: useful; specific fluid or remaining-volume claims: '
       'unverified until reservoir geometry, pressure, production and rock-physics data are checked.\n\n'
       '**Public references (metadata and original summaries, not imported full papers)**\n'+citations)


def search_public_crossref_metadata(topic, rows=12, timeout=8):
    """Opt-in PUBLIC metadata query; never sends user-entered proprietary question/data."""
    if topic not in TOPIC_CHOICES:raise ValueError('Only public generic topics permitted')
    query=TOPIC_CHOICES[topic]
    params=urlencode({'query.bibliographic':query,'rows':max(1,min(int(rows),25)),
                      'select':'DOI,title,published,container-title,URL,type'})
    url='https://api.crossref.org/works?'+params
    req=Request(url,headers={'User-Agent':'ANGSI-JR-geoscience-metadata/1.0 (public research)'})
    with urlopen(req,timeout=timeout) as response:
        message=json.load(response)['message']
    results=[]
    for r in message.get('items',[]):
        title=(r.get('title') or [''])[0]
        doi=r.get('DOI','')
        if not title or not doi:continue
        yr=(r.get('published',{}).get('date-parts') or [[None]])[0][0]
        results.append(dict(title=title,year=yr,doi=doi,url='https://doi.org/'+doi,
                            journal=(r.get('container-title') or [''])[0]))
    return pd.DataFrame(results)
