# -*- coding: utf-8 -*-
"""Turkish AI-Generated Text Detection Study V2 core utilities.

Primary design:
- 1000 matched ABSTRACT pairs -> retrospective 5x4 pair-aware nested CV.
- NEWSPAPER is never accessed from this module.
- Document-level raw features can be cached once.
- Any transformation learned from data is fit inside the relevant training partition.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from bisect import bisect_right
import hashlib, json, math, os, random, re, unicodedata
import numpy as np
import pandas as pd

TURKISH_MAP = str.maketrans({'I':'ı','İ':'i'})
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+(?:['’][A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+)?")
MORPH_WORD_RE = WORD_RE

PRIMARY_FEATURES = [
    'hapax_ratio_robust_z',
    'sentence_length_cv_robust_z',
    'stopword_density_robust_z',
    'conjunction_density_robust_z',
    'bigram_entropy_robust_z',
    'a1_morph_ttr_z',
    'a2_suffix_entropy_z',
    'a3_surface_vowel_profile_kl_z',
    'a4_fiilimsi_diversity_z',
    'a4_fiilimsi_density_z',
    'a4_fiilimsi_entropy_z',
    'c1_log_ppl_z',
    'c2_wordpiece_logprob_variance_z',
]
M2_FEATURES = PRIMARY_FEATURES[:5] + PRIMARY_FEATURES[-2:]
SURFACE_FEATURES = [
    'surface_digit_per100','surface_paren_per100','surface_percent_per100',
    'surface_curly_apostrophe_per100','surface_ascii_apostrophe_per100',
    'surface_colon_per100','surface_slash_per100','surface_semicolon_per100',
    'surface_hyphen_per100','surface_comma_per100','surface_curly_quote_per100',
    'surface_upper_abbrev_per100'
]

DEFAULT_STOPWORDS = {
'acaba','ama','aslında','az','bazı','belki','biri','birkaç','birşey','biz','bu','çok','çünkü','da','daha','de','defa','diye','eğer','en','gibi','hem','hep','hepsi','her','hiç','için','ile','ise','kez','ki','kim','mı','mu','mü','nasıl','ne','neden','nerde','nerede','nereye','niçin','niye','o','sanki','şey','siz','şu','tüm','ve','veya','ya','yani'
}
DEFAULT_CONJUNCTIONS = {'ve','veya','yahut','ya','ile','ama','ancak','fakat','lakin','çünkü','zira','oysa','halbuki','ki','hem','ne','gerek','ister'}


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def stable_seed(seed:int, *parts)->int:
    s='|'.join(map(str,(seed,)+parts)).encode('utf-8')
    return int(hashlib.sha256(s).hexdigest()[:8],16)


def turkish_lower(text):
    return str(text).translate(TURKISH_MAP).lower()


def normalize_text(text):
    """Minimal technical normalization; preserves case/punctuation/style."""
    s=unicodedata.normalize('NFC', '' if pd.isna(text) else str(text))
    s=''.join(ch if (ch in '\n\t' or unicodedata.category(ch) not in {'Cc','Cf','Cs','Co','Cn'}) else ' ' for ch in s)
    s=s.replace('\r\n','\n').replace('\r','\n')
    s=re.sub(r'[ \t]+',' ',s)
    s=re.sub(r'\n+',' ',s)
    return s.strip()


STRUCTURED_HEADING_RE = re.compile(
    r'(^|(?<=[.!?])\s+)(?:Amaç|Amaçlar|Giriş|Gereç ve Yöntem(?:ler)?|Yöntem(?:ler)?|Materyal ve Metot|Bulgular|Sonuç(?:lar)?|Tartışma)\s*:\s*',
    flags=re.IGNORECASE
)

def artifact_neutral_context_text(text):
    """Condition-B context for Turkish AI Detection M1/M2.

    Applies only the protocol-level typography normalization and structured-heading
    removal. Digit-containing lexical context is deliberately retained: Turkish AI Detection's
    sentence/TBS restrictions are applied at feature/scoring time rather than by
    deleting context from the document.
    """
    s=normalize_text(text)
    s=s.translate(str.maketrans({'’':"'",'‘':"'",'‛':"'",'“':'"','”':'"','„':'"','–':'-','—':'-'}))
    s=STRUCTURED_HEADING_RE.sub(lambda m: (m.group(1) or ''), s)
    return re.sub(r'\s+',' ',s).strip()


def artifact_neutral_text(text):
    """Condition-B text for M3/M4/M5.

    Starts from the context-preserving Condition-B normalization, then removes
    digit-containing whitespace tokens and numeric/symbol-only parenthetical
    groups exactly for the models whose protocol calls for that removal.
    """
    s=artifact_neutral_context_text(text)
    toks=[]
    for tok in re.findall(r'\S+', s):
        if any(ch.isdecimal() for ch in tok):
            continue
        toks.append(tok)
    s=' '.join(toks)
    s=re.sub(r'\(\s*[^A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]*\s*\)',' ',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s


def contains_letter(token):
    """Return True when a whitespace token contains at least one Unicode letter."""
    return any(ch.isalpha() for ch in str(token))


def letter_word_spans(text):
    """Whitespace-token spans containing at least one letter, in source coordinates."""
    s=str(text)
    return [(m.start(),m.end()) for m in re.finditer(r'\S+',s) if contains_letter(m.group(0))]


def wordpiece_positions_overlapping_spans(offsets,spans):
    """Select token positions whose character offsets overlap any eligible word span."""
    out=[]
    j=0
    spans=list(spans)
    for i,(a,b) in enumerate(offsets):
        a=int(a); b=int(b)
        if b<=a:
            continue
        while j<len(spans) and spans[j][1]<=a:
            j+=1
        if j<len(spans) and a<spans[j][1] and b>spans[j][0]:
            out.append(i)
    return out


def lexical_tokens(text):
    return [turkish_lower(t) for t in WORD_RE.findall(str(text))]


def surface_features(text):
    s=str(text); denom=max(len(s.split()),1)/100.0
    vals={
      'surface_digit_per100':sum(ch.isdecimal() for ch in s)/denom,
      'surface_paren_per100':(s.count('(')+s.count(')'))/denom,
      'surface_percent_per100':s.count('%')/denom,
      'surface_curly_apostrophe_per100':sum(s.count(x) for x in ['’','‘','‛'])/denom,
      'surface_ascii_apostrophe_per100':s.count("'")/denom,
      'surface_colon_per100':s.count(':')/denom,
      'surface_slash_per100':s.count('/')/denom,
      'surface_semicolon_per100':s.count(';')/denom,
      'surface_hyphen_per100':sum(s.count(x) for x in ['-','–','—'])/denom,
      'surface_comma_per100':s.count(',')/denom,
      'surface_curly_quote_per100':sum(s.count(x) for x in ['“','”','„'])/denom,
      'surface_upper_abbrev_per100':len(re.findall(r'\b(?:[A-ZÇĞİÖŞÜ]{2,})(?:\b|\d)',s))/denom,
    }
    return vals


def surface_model_matrix(df):
    """Protocol-defined M3 design matrix: 12 per-100 counts transformed by log1p."""
    X=df[SURFACE_FEATURES].apply(pd.to_numeric,errors='coerce').astype(float)
    if X.isna().any().any() or (X.to_numpy()<0).any():
        raise ValueError('M3 surface features must be finite non-negative counts/rates before log1p')
    return np.log1p(X)


def statistical_style_raw(text, mean_sentence_length, std_sentence_length, stopwords=None, conjunctions=None):
    stopwords=DEFAULT_STOPWORDS if stopwords is None else set(stopwords)
    conjunctions=DEFAULT_CONJUNCTIONS if conjunctions is None else set(conjunctions)
    tokens=lexical_tokens(text); n=len(tokens); cnt=Counter(tokens)
    hapax=sum(v==1 for v in cnt.values())
    bigrams=list(zip(tokens[:-1],tokens[1:])); bc=Counter(bigrams)
    ent=0.0
    if bigrams:
        N=len(bigrams)
        ent=-sum((c/N)*math.log2(c/N) for c in bc.values())
    mean=float(mean_sentence_length or 0); std=float(std_sentence_length or 0)
    return {
      'lexical_token_count':n,
      'unique_word_count':len(cnt),
      'ttr':len(cnt)/n if n else 0.0,
      'hapax_count':hapax,
      'hapax_ratio':hapax/n if n else 0.0,
      'sentence_length_cv':std/mean if mean>0 else 0.0,
      'bigram_entropy':ent,
      'stopword_count':sum(t in stopwords for t in tokens),
      'stopword_density':sum(t in stopwords for t in tokens)/n if n else 0.0,
      'conjunction_count':sum(t in conjunctions for t in tokens),
      'conjunction_density':sum(t in conjunctions for t in tokens)/n if n else 0.0,
    }

# ---------------- sentence segmentation ----------------
CONTINUATION_ABBREVIATIONS={'vb','vs','vd','ark','sp','spp','dak','sn','örn','bkz','md','m','no'}
TITLE_ABBREVIATIONS={'dr','prof','doç','yrd','arş','gör','uzm'}
TERMINAL_PUNCTUATION={'.','!','?','…'}
CLOSING_CHARS={'"',"'",'”','’',')',']','}'}

def next_nonspace_char(text,pos):
    m=re.search(r'\S',text[pos:]); return None if m is None else text[pos+m.start()]

def protect_academic_periods(text):
    text=str(text); chars=list(text)
    def mask(i):
        if 0<=i<len(chars): chars[i]=','
    for m in re.finditer(r'(?<=\d)\.(?=\d)',text): mask(m.start())
    for m in re.finditer(r'(?<=\S)\.(?=\d)',text): mask(m.start())
    for m in re.finditer(r'\.(?=\s*[\)\],;:\-])',text): mask(m.start())
    abbr=CONTINUATION_ABBREVIATIONS|TITLE_ABBREVIATIONS
    pat=re.compile(r'\b(?:'+'|'.join(sorted(map(re.escape,abbr),key=len,reverse=True))+r')\.',re.I)
    for m in pat.finditer(text):
        dot=m.end()-1; a=text[m.start():dot].lower(); nxt=next_nonspace_char(text,dot+1)
        if a in TITLE_ABBREVIATIONS:
            if nxt is not None: mask(dot)
        elif nxt is not None and (nxt.islower() or nxt.isdigit() or nxt in '([{'): mask(dot)
    for m in re.finditer(r'\b[A-ZÇĞİÖŞÜ]\.',text):
        dot=m.end()-1; nxt=next_nonspace_char(text,dot+1)
        if nxt is not None and (nxt.isalpha() or nxt in '([{'): mask(dot)
    out=''.join(chars); assert len(out)==len(text); return out

def is_plausible_sentence_start(text,start):
    if start<=0: return True
    if text[start-1].isspace() or text[start-1] in TERMINAL_PUNCTUATION: return True
    j=start-1
    while j>=0 and text[j] in CLOSING_CHARS: j-=1
    return j>=0 and text[j] in TERMINAL_PUNCTUATION

def sentence_features_v6(text,nlp,letter_words_only=False):
    text=str(text); protected=protect_academic_periods(text); doc=nlp(protected)
    raw=sorted(set(s.tokens[0].start_char for s in doc.sentences if s.tokens))
    starts=[]; removed=[]
    for st in raw:
        (starts if is_plausible_sentence_start(text,st) else removed).append(st)
    if not starts or starts[0]!=0: starts.insert(0,0)
    starts=sorted(set(starts)); spans=[]
    for i,st in enumerate(starts): spans.append((st, starts[i+1] if i+1<len(starts) else len(text)))
    lengths=[0]*len(spans)
    for m in re.finditer(r'\S+',text):
        if letter_words_only and not contains_letter(m.group(0)):
            continue
        idx=max(0,min(bisect_right(starts,m.start())-1,len(lengths)-1)); lengths[idx]+=1
    sent_texts=[text[a:b].strip() for a,b in spans]
    return {'sentence_count':len(lengths),'sentence_word_count':int(sum(lengths)),
            'mean_sentence_length':float(np.mean(lengths)) if lengths else 0.0,
            'std_sentence_length':float(np.std(lengths,ddof=0)) if lengths else 0.0,
            'min_sentence_length':int(np.min(lengths)) if lengths else 0,
            'max_sentence_length':int(np.max(lengths)) if lengths else 0,
            'sentence_lengths':json.dumps(lengths,ensure_ascii=False),
            'sentence_texts':json.dumps(sent_texts,ensure_ascii=False),
            'sentence_spans':json.dumps(spans),
            'removed_false_boundary_count':len(removed),
            'removed_false_boundaries':json.dumps(removed)}

# ---------------- morphology ----------------
NONFINITE_EXACT_TAGS={'ByDoingSo','AfterDoingSo','WithoutDoingSo','WithoutHavingDoneSo','While','When','AsIf','AsLongAs'}
BACK_VOWELS=set('aıou'); FRONT_VOWELS=set('eiöü'); ALL_VOWELS=BACK_VOWELS|FRONT_VOWELS
VOWEL_NORMALIZATION=str.maketrans({'â':'a','î':'i','û':'u','Â':'a','Î':'i','Û':'u'})

def shannon_entropy(counter):
    total=sum(counter.values());
    if total<=0:return 0.0
    return -sum((v/total)*math.log2(v/total) for v in counter.values() if v>0)

def is_nonfinite_tag(tag): return tag.startswith('Inf') or tag.endswith('Part') or tag in NONFINITE_EXACT_TAGS

def normalize_vowels(word): return turkish_lower(word).translate(VOWEL_NORMALIZATION)

def vowel_harmony_class(word):
    v=[c for c in normalize_vowels(word) if c in ALL_VOWELS]
    if len(v)<2:return 'insufficient'
    b=any(x in BACK_VOWELS for x in v); f=any(x in FRONT_VOWELS for x in v)
    if b and not f:return 'back_harmonic'
    if f and not b:return 'front_harmonic'
    return 'mixed'

class ZeyrekFeatureExtractor:
    def __init__(self):
        import zeyrek
        self.analyzer=zeyrek.MorphAnalyzer(); self.cache={}
    def analyze(self,token):
        if token in self.cache:return self.cache[token]
        try: result=self.analyzer.analyze(token)
        except Exception: result=[]
        if not result: out=()
        else:
            candidates=result[0] if isinstance(result[0],list) else result
            valid=[]
            for p in candidates:
                if getattr(p,'lemma',None) not in (None,'Unk') and getattr(p,'pos',None) not in (None,'Unk'): valid.append(p)
            unique={}
            for p in valid:
                sig=self.morph_signature(p); unique.setdefault(sig,p)
            out=tuple(unique.values())
        self.cache[token]=out; return out
    @staticmethod
    def realized(parse):
        formatted=getattr(parse,'formatted','')
        if '] ' not in formatted:return []
        body=formatted.split('] ',1)[1]; pieces=re.split(r'[+|]',body); out=[]
        for i,piece in enumerate(pieces):
            left=piece.split('→',1)[0]
            if ':' not in left:continue
            surface,tag=left.rsplit(':',1); surface=surface.strip(); tag=tag.strip()
            if i==0:continue
            if surface and tag:out.append((surface,tag))
        return out
    def morph_signature(self,parse):
        return (tuple(getattr(parse,'morphemes',()) or ()), tuple(t for _,t in self.realized(parse)))
    @staticmethod
    def token_type(analyses):
        return tuple(sorted(set(tuple(getattr(p,'morphemes',()) or ()) for p in analyses),key=str))
    def nonfinite_fractional(self,analyses):
        k=len(analyses)
        if not k:return 0.0,Counter()
        w=1/k; mass=0.; patterns=Counter()
        for p in analyses:
            tags=tuple(sorted(set(t for t in (getattr(p,'morphemes',()) or ()) if is_nonfinite_tag(t))))
            if tags: mass+=w; patterns[tags]+=w
        return mass,patterns
    def extract(self,text):
        tokens=[turkish_lower(x) for x in MORPH_WORD_RE.findall(str(text))]
        total=len(tokens); analyzed=0; ambiguous=0; candidates_total=0
        morph_types=[]; suffix_tags=Counter(); suffix_surface=Counter(); fiilimsi_mass=0.; fiil_patterns=Counter(); poss=[]
        vcounts=Counter(vowel_harmony_class(t) for t in tokens)
        for tok in tokens:
            analyses=self.analyze(tok); k=len(analyses)
            if not k:continue
            analyzed+=1; candidates_total+=k; ambiguous+=int(k>1); morph_types.append(self.token_type(analyses)); w=1/k
            for p in analyses:
                for surface,tag in self.realized(p): suffix_tags[tag]+=w; suffix_surface[surface]+=w
            mass,pats=self.nonfinite_fractional(analyses); fiilimsi_mass+=mass; fiil_patterns.update(pats)
            cp=tuple(sorted(pats.keys(),key=str))
            if cp: poss.append(cp)
        eligible=vcounts['back_harmonic']+vcounts['front_harmonic']+vcounts['mixed']
        br=vcounts['back_harmonic']/eligible if eligible else 0.; fr=vcounts['front_harmonic']/eligible if eligible else 0.; mr=vcounts['mixed']/eligible if eligible else 0.
        fent=shannon_entropy(fiil_patterns)
        return {
          'morph_token_count':total,'morph_analyzed_tokens':analyzed,'morph_unknown_tokens':total-analyzed,
          'morph_analysis_coverage':analyzed/total if total else 0.,'morph_ambiguous_tokens':ambiguous,
          'morph_ambiguity_rate':ambiguous/analyzed if analyzed else 0.,'mean_morph_candidates':candidates_total/analyzed if analyzed else 0.,
          'mes_analyzed_token_count':analyzed,'morph_type_count':len(set(morph_types)),
          'morph_ttr':len(set(morph_types))/analyzed if analyzed else 0.,
          'suffix_tag_entropy':shannon_entropy(suffix_tags),'suffix_surface_entropy':shannon_entropy(suffix_surface),
          'suffix_tag_type_count':len(suffix_tags),'suffix_surface_type_count':len(suffix_surface),
          'fiilimsi_expected_mass':fiilimsi_mass,'fiilimsi_expected_density':fiilimsi_mass/analyzed if analyzed else 0.,
          'fiilimsi_diversity_ratio':len(set(poss))/len(poss) if poss else 0.,
          'fiilimsi_pattern_entropy':fent,'fiilimsi_effective_patterns':2**fent if fiil_patterns else 0.,
          'vowel_harmony_eligible_count':eligible,'vowel_harmony_eligible_ratio':eligible/total if total else 0.,
          'vowel_back_ratio':br,'vowel_front_ratio':fr,'vowel_mixed_ratio':mr,
        }

# ---------------- BERT PLL ----------------
class BertPLLScorer:
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased', device=None, revision=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.torch=torch; self.model_name=model_name
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision)
        self.model=AutoModelForMaskedLM.from_pretrained(model_name,revision=revision)
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.model.to(self.device).eval()
        self.commit_hash=getattr(self.model.config,'_commit_hash',None) or getattr(self.tokenizer,'_commit_hash',None)
        self.max_sequence=int(getattr(self.model.config,'max_position_embeddings',512)); self.max_content=self.max_sequence-2
    def windows(self,n,stride=None):
        stride=stride or self.max_content//2
        if n<=self.max_content:return [(0,n)]
        starts=list(range(0,n-self.max_content+1,stride)); last=n-self.max_content
        if starts[-1]!=last:starts.append(last)
        return [(s,min(s+self.max_content,n)) for s in starts]
    @staticmethod
    def owners(n,windows):
        owners=[[] for _ in windows]
        for i in range(n):
            cand=[]
            for wi,(a,b) in enumerate(windows):
                if a<=i<b:cand.append((min(i-a,b-1-i),wi))
            if not cand:raise RuntimeError(f'Unowned token {i}')
            owners[max(cand)[1]].append(i)
        assert sorted(x for g in owners for x in g)==list(range(n)); return owners
    def _score_positions(self,text,positions=None,mask_batch_size=32,offsets=None):
        torch=self.torch
        if offsets is None:
            enc=self.tokenizer(str(text),add_special_tokens=False,truncation=False)
            ids=enc['input_ids']
        else:
            enc=self.tokenizer(str(text),add_special_tokens=False,truncation=False,return_offsets_mapping=True)
            ids=enc['input_ids']; offsets=enc['offset_mapping']
        n=len(ids)
        target_positions=list(range(n)) if positions is None else sorted(set(int(i) for i in positions if 0<=int(i)<n))
        if n==0 or not target_positions:
            return {'bert_wordpiece_count':n,'pll_scored_wordpiece_count':0,'pll_window_count':0,'pseudo_log_likelihood_mean':np.nan,'pseudo_perplexity':np.nan,'wordpiece_logprob_variance':np.nan,'wordpiece_logprob_std':np.nan}
        target_set=set(target_positions)
        wins=self.windows(n); owners=self.owners(n,wins); vals={}
        used_windows=0
        for (a,b),owned_all in zip(wins,owners):
            owned=[g for g in owned_all if g in target_set]
            if not owned:continue
            used_windows+=1
            content=ids[a:b]; seq=[self.tokenizer.cls_token_id]+content+[self.tokenizer.sep_token_id]
            base=torch.tensor(seq,dtype=torch.long,device=self.device)
            local=[1+(g-a) for g in owned]; targets=[ids[g] for g in owned]
            for bs in range(0,len(local),mask_batch_size):
                loc=local[bs:bs+mask_batch_size]; tar=targets[bs:bs+mask_batch_size]; glob=owned[bs:bs+mask_batch_size]; B=len(loc)
                x=base.unsqueeze(0).expand(B,-1).clone(); rows=torch.arange(B,device=self.device); mp=torch.tensor(loc,device=self.device); ti=torch.tensor(tar,device=self.device)
                x[rows,mp]=self.tokenizer.mask_token_id; att=torch.ones_like(x)
                with torch.inference_mode():
                    if self.device.startswith('cuda'):
                        with torch.autocast(device_type='cuda',dtype=torch.float16): logits=self.model(input_ids=x,attention_mask=att).logits[rows,mp]
                    else: logits=self.model(input_ids=x,attention_mask=att).logits[rows,mp]
                    lp=torch.log_softmax(logits.float(),dim=-1)[rows,ti]
                for g,v in zip(glob,lp.detach().cpu().numpy()): vals[int(g)]=float(v)
        arr=np.asarray([vals.get(i,np.nan) for i in target_positions],dtype=float)
        if not np.isfinite(arr).all():raise RuntimeError('Missing PLL scores for selected WordPieces')
        m=float(arr.mean()); v=float(arr.var()); sd=float(arr.std())
        return {'bert_wordpiece_count':n,'pll_scored_wordpiece_count':len(target_positions),'pll_window_count':used_windows,'pseudo_log_likelihood_mean':m,'pseudo_perplexity':float(np.exp(-m)),'wordpiece_logprob_variance':v,'wordpiece_logprob_std':sd}

    def score(self,text,mask_batch_size=32):
        return self._score_positions(text,positions=None,mask_batch_size=mask_batch_size)

    def score_letter_wordpieces(self,text,mask_batch_size=32):
        """PLL over WordPieces belonging to letter-containing whitespace words.

        The full Condition-B text is tokenized and retained as model context; only
        the masked/scored WordPiece targets are restricted by source-text spans.
        """
        enc=self.tokenizer(str(text),add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        ids=enc['input_ids']; offsets=enc.get('offset_mapping')
        if offsets is None:
            raise RuntimeError('Fast-tokenizer offset mapping is required for exact Condition-B TBS scoring')
        positions=wordpiece_positions_overlapping_spans(offsets,letter_word_spans(text))
        # Avoid a second offset-mapping path inside _score_positions; token IDs are
        # deterministic for the same tokenizer/text, and positions index those IDs.
        return self._score_positions(text,positions=positions,mask_batch_size=mask_batch_size)

# ---------------- folds ----------------
def create_nested_folds(base_df,seed=42):
    pair=base_df.loc[base_df.label.eq('ai'),['pair_id','model','field','broad_field']].drop_duplicates().copy()
    assert len(pair)==1000 and pair.pair_id.is_unique
    pair['outer_fold']=-1
    for (model,field),idx in pair.groupby(['model','field'],sort=True).groups.items():
        ids=pair.loc[list(idx),'pair_id'].tolist(); assert len(ids)==10
        rng=np.random.default_rng(stable_seed(seed,'outer',model,field)); ids=list(rng.permutation(ids))
        for pos,pid in enumerate(ids): pair.loc[pair.pair_id.eq(pid),'outer_fold']=pos//2
    for outer in range(5):
        col=f'inner_fold_outer{outer}'; pair[col]=-1
        for (model,field),g in pair.groupby(['model','field'],sort=True):
            ids=g.loc[g.outer_fold.ne(outer),'pair_id'].tolist(); assert len(ids)==8
            rng=np.random.default_rng(stable_seed(seed,'inner',outer,model,field)); ids=list(rng.permutation(ids))
            for pos,pid in enumerate(ids): pair.loc[pair.pair_id.eq(pid),col]=pos//2
    assert set(pair.outer_fold)==set(range(5))
    return pair.sort_values('pair_id').reset_index(drop=True)

# ---------------- train-only feature transform ----------------
def _robust_params(s):
    s=pd.Series(s,dtype=float).dropna(); med=float(s.median()); mad=float((s-med).abs().median()); scale=1.4826*mad
    if not np.isfinite(scale) or scale<=0: scale=float(s.std(ddof=0))
    if not np.isfinite(scale) or scale<=0: scale=1.0
    return med,scale

def _kl(p,q,eps=1e-6):
    p=np.asarray(p,float)+eps; q=np.asarray(q,float)+eps; p/=p.sum(); q/=q.sum(); return float(np.sum(p*np.log2(p/q)))

class TurkishAIDetectionFeatureTransformer:
    """Fits all human-reference nuisance/reference transforms on the provided training partition only."""
    def __init__(self): self.fitted=False
    def fit(self,df):
        from sklearn.linear_model import LinearRegression
        h=df.loc[df.label.eq('human')].copy(); assert len(h)>0
        self.residual_models={}
        for key,target,size in [
            ('bigram','bigram_entropy','lexical_token_count'),('a1','morph_ttr','mes_analyzed_token_count'),
            ('a2','suffix_tag_entropy','mes_analyzed_token_count'),('a4div','fiilimsi_diversity_ratio','fiilimsi_expected_mass')]:
            X=np.log1p(pd.to_numeric(h[size],errors='coerce').fillna(0).to_numpy(float)).reshape(-1,1); y=pd.to_numeric(h[target],errors='coerce').to_numpy(float)
            m=LinearRegression().fit(X,y); self.residual_models[key]=(m,target,size)
        q=h[['vowel_back_ratio','vowel_front_ratio','vowel_mixed_ratio']].mean().to_numpy(float); self.vowel_ref=q/q.sum()
        tmp=self._intermediate(h)
        self.z_params={}
        mapping={
          'hapax_ratio_robust_z':'hapax_ratio','sentence_length_cv_robust_z':'sentence_length_cv','stopword_density_robust_z':'stopword_density','conjunction_density_robust_z':'conjunction_density','bigram_entropy_robust_z':'bigram_resid',
          'a1_morph_ttr_z':'a1_resid','a2_suffix_entropy_z':'a2_resid','a3_surface_vowel_profile_kl_z':'surface_vowel_kl','a4_fiilimsi_diversity_z':'a4div_resid','a4_fiilimsi_density_z':'fiilimsi_expected_density','a4_fiilimsi_entropy_z':'fiilimsi_pattern_entropy','c1_log_ppl_z':'log_pseudo_perplexity','c2_wordpiece_logprob_variance_z':'wordpiece_logprob_variance'
        }
        for out,src in mapping.items(): self.z_params[out]=(src,)+_robust_params(tmp[src])
        self.fitted=True; return self
    def _intermediate(self,df):
        out=df.copy()
        for key,(m,target,size) in self.residual_models.items():
            X=np.log1p(pd.to_numeric(out[size],errors='coerce').fillna(0).to_numpy(float)).reshape(-1,1)
            out[f'{key}_resid']=pd.to_numeric(out[target],errors='coerce').to_numpy(float)-m.predict(X)
        out['surface_vowel_kl']=[_kl(r,self.vowel_ref) for r in out[['vowel_back_ratio','vowel_front_ratio','vowel_mixed_ratio']].to_numpy(float)]
        return out
    def transform(self,df):
        if not self.fitted:raise RuntimeError('Transformer not fit')
        tmp=self._intermediate(df); X=pd.DataFrame(index=df.index)
        for out,(src,med,scale) in self.z_params.items(): X[out]=(pd.to_numeric(tmp[src],errors='coerce')-med)/scale
        if X.isna().any().any(): raise ValueError('NaN in transformed features: '+str(X.columns[X.isna().any()].tolist()))
        return X
    def fit_transform(self,df): return self.fit(df).transform(df)

# ---------------- modeling helpers ----------------
def make_logreg(C):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([('scaler',StandardScaler()),('lr',LogisticRegression(C=float(C),penalty='l2',solver='liblinear',max_iter=5000,random_state=42))])

def fit_platt(scores,y):
    from sklearn.linear_model import LogisticRegression
    m=LogisticRegression(penalty=None,solver='lbfgs',max_iter=5000); m.fit(np.asarray(scores).reshape(-1,1),np.asarray(y)); return m

def platt_prob(cal,scores): return cal.predict_proba(np.asarray(scores).reshape(-1,1))[:,1]

def order_stat_threshold(human_probs,k_exceed):
    x=np.sort(np.asarray(human_probs,float))[::-1]
    if len(x)<=k_exceed: raise ValueError('Not enough human scores')
    return float(x[k_exceed])  # (k_exceed+1)-th highest; classify strictly > tau

def y_from_label(s): return np.where(pd.Series(s).astype(str).eq('ai'),1,0)

def standardized_pauc(y,score,max_fpr=.03):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,score,max_fpr=max_fpr))

def evaluate_binary(y,prob,pred=None,max_fpr=.03):
    from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss,confusion_matrix
    y=np.asarray(y,int); prob=np.asarray(prob,float)
    d={'auc':float(roc_auc_score(y,prob)),'std_pauc_003':float(roc_auc_score(y,prob,max_fpr=max_fpr)),'brier':float(brier_score_loss(y,prob)),'log_loss':float(log_loss(y,prob,labels=[0,1]))}
    if pred is not None:
        tn,fp,fn,tp=confusion_matrix(y,np.asarray(pred,int),labels=[0,1]).ravel(); d.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'fpr':fp/(fp+tn) if fp+tn else np.nan,'tpr':tp/(tp+fn) if tp+fn else np.nan})
    return d


def ece_equal_width(y,prob,n_bins=10):
    """Expected calibration error with fixed equal-width probability bins.

    Bins are [0, 1/n), ..., [(n-1)/n, 1], and the statistic is the
    sample-weighted absolute difference between empirical positive rate and
    mean predicted probability in each non-empty bin.
    """
    y=np.asarray(y,int); prob=np.asarray(prob,float)
    if len(y)!=len(prob) or len(y)==0: raise ValueError('y/prob must be non-empty and aligned')
    if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any(): raise ValueError('probabilities must be finite in [0,1]')
    edges=np.linspace(0.0,1.0,int(n_bins)+1)
    idx=np.digitize(prob,edges[1:-1],right=False)
    total=len(y); ece=0.0
    for b in range(int(n_bins)):
        mask=idx==b
        if not mask.any(): continue
        ece += (mask.sum()/total)*abs(float(y[mask].mean())-float(prob[mask].mean()))
    return float(ece)


def clopper_pearson_interval(k,n,alpha=.05):
    """Two-sided exact Clopper-Pearson interval for a binomial proportion."""
    from scipy.stats import beta
    k=int(k); n=int(n)
    if n<=0 or k<0 or k>n: raise ValueError('require 0 <= k <= n and n > 0')
    lo=0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi=1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi


def paired_pair_bootstrap_auc_delta(m1_oof,m2_oof,B=2000,seed=42):
    """Paired pair-ID percentile bootstrap for AUC(M1)-AUC(M2).

    Each bootstrap draw samples pair IDs with replacement and carries both the
    human and matched AI row together. M1/M2 predictions must refer to the same
    sample/pair/label rows.
    """
    from sklearn.metrics import roc_auc_score
    req={'sample_id','pair_id','label','prob'}
    for name,df in [('M1',m1_oof),('M2',m2_oof)]:
        miss=req.difference(df.columns)
        if miss: raise ValueError(f'{name} missing columns: {sorted(miss)}')
        if df.sample_id.isna().any() or not df.sample_id.is_unique: raise ValueError(f'{name} sample_id must be unique')
    a=m1_oof[['sample_id','pair_id','label','prob']].rename(columns={'prob':'prob_m1'}).copy()
    b=m2_oof[['sample_id','pair_id','label','prob']].rename(columns={'prob':'prob_m2'}).copy()
    m=a.merge(b,on='sample_id',how='inner',validate='one_to_one',suffixes=('_a','_b'))
    if len(m)!=len(a) or len(m)!=len(b): raise ValueError('M1/M2 sample sets differ')
    if not m.pair_id_a.astype(str).eq(m.pair_id_b.astype(str)).all() or not m.label_a.astype(str).eq(m.label_b.astype(str)).all():
        raise ValueError('M1/M2 pair_id or label alignment differs')
    m=m.rename(columns={'pair_id_a':'pair_id','label_a':'label'}).drop(columns=['pair_id_b','label_b'])
    pair_sizes=m.groupby('pair_id').size()
    if len(pair_sizes)==0 or not pair_sizes.eq(2).all(): raise ValueError('each pair_id must contribute exactly two rows')
    if not m.groupby('pair_id').label.apply(lambda x:set(map(str,x))=={'human','ai'}).all():
        raise ValueError('each pair must contain one human and one ai row')
    y=y_from_label(m.label); point=float(roc_auc_score(y,m.prob_m1)-roc_auc_score(y,m.prob_m2))
    pair_ids=np.asarray(sorted(pair_sizes.index,key=str),dtype=object)
    row_groups={pid:np.flatnonzero(m.pair_id.to_numpy()==pid) for pid in pair_ids}
    rng=np.random.default_rng(int(seed)); draws=np.empty(int(B),dtype=float)
    for i in range(int(B)):
        sampled=rng.choice(pair_ids,size=len(pair_ids),replace=True)
        rows=np.concatenate([row_groups[pid] for pid in sampled])
        yy=y[rows]
        draws[i]=roc_auc_score(yy,m.prob_m1.to_numpy()[rows])-roc_auc_score(yy,m.prob_m2.to_numpy()[rows])
    lo,hi=np.percentile(draws,[2.5,97.5])
    return {
        'delta_auc':point,'ci95_lower':float(lo),'ci95_upper':float(hi),
        'B':int(B),'seed':int(seed),'ci_method':'percentile_pair_bootstrap',
        'bootstrap_deltas':draws,
    }


def paired_pair_bootstrap_auc_tpr_delta(reference_oof, comparator_oof, B=2000, seed=42):
    """Pair-ID bootstrap for AUC and operating-point TPR differences.

    Differences are reference minus comparator. Each sampled pair carries its
    human and matched AI observations, preserving the protocol's paired unit.
    """
    from sklearn.metrics import roc_auc_score
    required={'sample_id','pair_id','label','prob','pred'}
    frames=[]
    for name,df in [('reference',reference_oof),('comparator',comparator_oof)]:
        missing=required.difference(df.columns)
        if missing: raise ValueError(f'{name} missing columns: {sorted(missing)}')
        if df.sample_id.isna().any() or not df.sample_id.is_unique:
            raise ValueError(f'{name} sample_id must be unique')
        frames.append(df[['sample_id','pair_id','label','prob','pred']].copy())
    left=frames[0].rename(columns={'prob':'prob_reference','pred':'pred_reference'})
    right=frames[1].rename(columns={'prob':'prob_comparator','pred':'pred_comparator'})
    merged=left.merge(right,on='sample_id',how='inner',validate='one_to_one',suffixes=('_left','_right'))
    if len(merged)!=len(left) or len(merged)!=len(right): raise ValueError('comparison sample sets differ')
    if not merged.pair_id_left.astype(str).eq(merged.pair_id_right.astype(str)).all():
        raise ValueError('comparison pair_id alignment differs')
    if not merged.label_left.astype(str).eq(merged.label_right.astype(str)).all():
        raise ValueError('comparison label alignment differs')
    merged=merged.rename(columns={'pair_id_left':'pair_id','label_left':'label'}).drop(columns=['pair_id_right','label_right'])
    pair_sizes=merged.groupby('pair_id').size()
    if len(pair_sizes)==0 or not pair_sizes.eq(2).all():
        raise ValueError('each pair_id must contribute exactly two rows')
    if not merged.groupby('pair_id').label.apply(lambda x:set(map(str,x))=={'human','ai'}).all():
        raise ValueError('each pair must contain one human and one ai row')
    y=y_from_label(merged.label)
    ai=y==1
    prob_ref=pd.to_numeric(merged.prob_reference).to_numpy(float)
    prob_cmp=pd.to_numeric(merged.prob_comparator).to_numpy(float)
    pred_ref=pd.to_numeric(merged.pred_reference).to_numpy(int)
    pred_cmp=pd.to_numeric(merged.pred_comparator).to_numpy(int)
    point_auc=float(roc_auc_score(y,prob_ref)-roc_auc_score(y,prob_cmp))
    point_tpr=float(pred_ref[ai].mean()-pred_cmp[ai].mean())
    pair_ids=np.asarray(sorted(pair_sizes.index,key=str),dtype=object)
    row_groups={pid:np.flatnonzero(merged.pair_id.to_numpy()==pid) for pid in pair_ids}
    rng=np.random.default_rng(int(seed))
    auc_draws=np.empty(int(B),dtype=float); tpr_draws=np.empty(int(B),dtype=float)
    for i in range(int(B)):
        sampled=rng.choice(pair_ids,size=len(pair_ids),replace=True)
        rows=np.concatenate([row_groups[pid] for pid in sampled])
        yy=y[rows]; aa=yy==1
        auc_draws[i]=roc_auc_score(yy,prob_ref[rows])-roc_auc_score(yy,prob_cmp[rows])
        tpr_draws[i]=pred_ref[rows][aa].mean()-pred_cmp[rows][aa].mean()
    auc_lo,auc_hi=np.percentile(auc_draws,[2.5,97.5])
    tpr_lo,tpr_hi=np.percentile(tpr_draws,[2.5,97.5])
    return {
        'delta_auc':point_auc,'delta_tpr':point_tpr,
        'auc_ci95_lower':float(auc_lo),'auc_ci95_upper':float(auc_hi),
        'tpr_ci95_lower':float(tpr_lo),'tpr_ci95_upper':float(tpr_hi),
        'B':int(B),'seed':int(seed),'ci_method':'percentile_pair_bootstrap',
        'bootstrap_delta_auc':auc_draws,'bootstrap_delta_tpr':tpr_draws,
    }


def protocol_year_block(values):
    """Map ABSTRACT years to the three prespecified S-04 blocks."""
    years=pd.to_numeric(pd.Series(values),errors='raise').astype(int)
    result=np.select(
        [years.le(2014), years.between(2015,2017), years.eq(2018)],
        ['le_2014','2015_2017','2018'], default='OUT_OF_PROTOCOL_RANGE'
    )
    if np.any(result=='OUT_OF_PROTOCOL_RANGE'):
        bad=sorted(set(years[result=='OUT_OF_PROTOCOL_RANGE'].tolist()))
        raise ValueError(f'year outside prespecified S-04 blocks: {bad}')
    return pd.Series(result,index=years.index,dtype='object')


def protocol_digit_density_bin(values):
    """Apply the pre-result IA-06 amendment's fixed per-100-word bins."""
    density=pd.to_numeric(pd.Series(values),errors='raise').astype(float)
    if not np.isfinite(density).all() or density.lt(0).any():
        raise ValueError('digit density must be finite and non-negative')
    labels=np.select(
        [density.eq(0), density.gt(0)&density.le(1), density.gt(1)],
        ['zero','gt_0_le_1','gt_1'], default='INVALID'
    )
    if np.any(labels=='INVALID'): raise ValueError('invalid digit-density bin')
    return pd.Series(labels,index=density.index,dtype='object')


def s04_human_score_linear_model(m1_oof, raw_features):
    """Fit the prespecified descriptive S-04 linear model on human OOF scores."""
    required_oof={'sample_id','label','prob'}; required_raw={'sample_id','year','broad_field','surface_digit_per100'}
    if required_oof.difference(m1_oof.columns): raise ValueError('M1 OOF lacks S-04 columns')
    if required_raw.difference(raw_features.columns): raise ValueError('raw features lack S-04 columns')
    merged=m1_oof[['sample_id','label','prob']].merge(
        raw_features[['sample_id','year','broad_field','surface_digit_per100']],
        on='sample_id',how='inner',validate='one_to_one'
    )
    if len(merged)!=len(m1_oof): raise ValueError('S-04 feature join is incomplete')
    human=merged.loc[merged.label.astype(str).eq('human')].copy()
    human['year_block']=protocol_year_block(human.year).to_numpy()
    human['log1p_number_density']=np.log1p(pd.to_numeric(human.surface_digit_per100).to_numpy(float))
    year_levels=['le_2014','2015_2017','2018']
    field_levels=sorted(human.broad_field.astype(str).unique().tolist())
    columns=[np.ones(len(human),dtype=float)]; names=['intercept']
    for level in year_levels[1:]:
        columns.append(human.year_block.eq(level).to_numpy(float)); names.append(f'year_block[{level}]')
    for level in field_levels[1:]:
        columns.append(human.broad_field.astype(str).eq(level).to_numpy(float)); names.append(f'broad_field[{level}]')
    columns.append(human.log1p_number_density.to_numpy(float)); names.append('log1p_number_density')
    design=np.column_stack(columns); response=pd.to_numeric(human.prob).to_numpy(float)
    coefficients,residuals,rank,singular=np.linalg.lstsq(design,response,rcond=None)
    fitted=design@coefficients; ss_res=float(np.square(response-fitted).sum()); ss_tot=float(np.square(response-response.mean()).sum())
    return {
        'analysis':'S04_human_calibrated_score_linear_model',
        'response':'M1_outer_oof_calibrated_probability_human_only',
        'n_human':int(len(human)),
        'year_reference':year_levels[0],
        'broad_field_reference':field_levels[0],
        'coefficients':{name:float(value) for name,value in zip(names,coefficients)},
        'rank':int(rank),
        'r_squared':float(1.0-ss_res/ss_tot) if ss_tot>0 else 0.0,
        'formula':'score ~ year_block + broad_field + log1p(number_density_per_100_words)',
    }


def human_fpr_descriptive_table(m1_oof, raw_features):
    """Return prespecified human FPR summaries by field, year block and density."""
    columns=['sample_id','year','field','broad_field','surface_digit_per100']
    merged=m1_oof[['sample_id','label','pred']].merge(
        raw_features[columns],on='sample_id',how='inner',validate='one_to_one'
    )
    if len(merged)!=len(m1_oof): raise ValueError('descriptive FPR feature join is incomplete')
    human=merged.loc[merged.label.astype(str).eq('human')].copy()
    human['year_block']=protocol_year_block(human.year).to_numpy()
    human['digit_density_bin']=protocol_digit_density_bin(human.surface_digit_per100).to_numpy()
    rows=[]
    for dimension in ('field','broad_field','year_block','digit_density_bin'):
        for level,group in human.groupby(dimension,sort=True):
            fp=int(pd.to_numeric(group.pred).astype(int).sum()); n=int(len(group))
            rows.append({'dimension':dimension,'level':str(level),'n_human':n,'false_positives':fp,'fpr':float(fp/n)})
    return pd.DataFrame(rows)


def condition_d_broad_field_metrics(digitfree_oof):
    """Compute no-refit broad-field AUC/FPR on the matched digit-free subset."""
    from sklearn.metrics import roc_auc_score
    required={'broad_field','label','prob','pred'}
    missing=required.difference(digitfree_oof.columns)
    if missing: raise ValueError(f'Condition-D OOF missing columns: {sorted(missing)}')
    rows=[]
    for level,group in digitfree_oof.groupby('broad_field',sort=True):
        y=y_from_label(group.label); human=y==0
        if len(np.unique(y))!=2: raise ValueError(f'Condition-D broad field lacks both labels: {level}')
        pred=pd.to_numeric(group.pred).to_numpy(int)
        rows.append({
            'broad_field':str(level),'n_rows':int(len(group)),'n_human':int(human.sum()),
            'n_ai':int((~human).sum()),'auc':float(roc_auc_score(y,pd.to_numeric(group.prob))),
            'false_positives':int(pred[human].sum()),'fpr':float(pred[human].mean()),
        })
    return pd.DataFrame(rows)


def p2_mes_decision(delta_auc,ci95_lower,ci95_upper,sesoi=.01):
    """Prespecified three-way MES incremental-contribution decision rule."""
    d=float(delta_auc); lo=float(ci95_lower); hi=float(ci95_upper); s=float(sesoi)
    if lo>0 and d>=s: return 'meaningful_incremental_contribution'
    if hi<s: return 'no_meaningful_incremental_contribution'
    return 'uncertain'

def _indices_for_pairs(df,pair_ids): return df.index[df.pair_id.isin(set(pair_ids))].to_numpy()

def _best_c(rows):
    rows=sorted(rows,key=lambda r:(-r['std_pauc'],r['C']))
    return float(rows[0]['C'])

def nested_cv_turkish_ai_detection(raw,folds,feature_set='M1',c_grid=(.01,.03,.1,.3,1,3,10),k_exceed=15,
                      collect_m05=False):
    """Nested CV for M1/M2, optionally retaining minimal M-05 M1 human scores."""
    from sklearn.metrics import roc_auc_score
    if collect_m05 and feature_set != 'M1':
        raise ValueError('M-05 diagnostic collection is defined only for M1')
    feats=PRIMARY_FEATURES if feature_set=='M1' else M2_FEATURES
    pred_rows=[]; fold_rows=[]; tune=[]; m05_rows=[]
    for outer in range(5):
        ft=folds; train_pairs=ft.loc[ft.outer_fold.ne(outer),'pair_id'].tolist(); test_pairs=ft.loc[ft.outer_fold.eq(outer),'pair_id'].tolist(); inner_col=f'inner_fold_outer{outer}'
        tr=raw.loc[raw.pair_id.isin(train_pairs)].copy(); te=raw.loc[raw.pair_id.isin(test_pairs)].copy(); ytr=y_from_label(tr.label); yte=y_from_label(te.label)
        crows=[]
        for C in c_grid:
            oof=np.full(len(tr),np.nan)
            for inner in range(4):
                val_pairs=ft.loc[(ft.outer_fold.ne(outer)) & (ft[inner_col].eq(inner)),'pair_id'].tolist(); mask=tr.pair_id.isin(val_pairs).to_numpy(); fit=~mask
                tf=TurkishAIDetectionFeatureTransformer().fit(tr.iloc[np.where(fit)[0]]); Xf=tf.transform(tr.iloc[np.where(fit)[0]])[feats]; Xv=tf.transform(tr.iloc[np.where(mask)[0]])[feats]
                mdl=make_logreg(C).fit(Xf,ytr[fit]); oof[mask]=mdl.decision_function(Xv)
            assert np.isfinite(oof).all(); p=standardized_pauc(ytr,oof); crows.append({'outer_fold':outer,'model':feature_set,'C':float(C),'std_pauc':p})
        tune.extend(crows); best=_best_c(crows)
        inner_dec=np.full(len(tr),np.nan)
        for inner in range(4):
            val_pairs=ft.loc[(ft.outer_fold.ne(outer)) & (ft[inner_col].eq(inner)),'pair_id'].tolist(); mask=tr.pair_id.isin(val_pairs).to_numpy(); fit=~mask
            tf=TurkishAIDetectionFeatureTransformer().fit(tr.iloc[np.where(fit)[0]]); Xf=tf.transform(tr.iloc[np.where(fit)[0]])[feats]; Xv=tf.transform(tr.iloc[np.where(mask)[0]])[feats]
            mdl=make_logreg(best).fit(Xf,ytr[fit]); inner_dec[mask]=mdl.decision_function(Xv)
        cal=fit_platt(inner_dec,ytr); inner_prob=platt_prob(cal,inner_dec); tau=order_stat_threshold(inner_prob[ytr==0],k_exceed)
        if collect_m05:
            pair_inner=ft.set_index('pair_id')[inner_col]
            human_positions=np.flatnonzero(ytr==0)
            for position in human_positions:
                row=tr.iloc[position]
                m05_rows.append({
                    'outer_fold':int(outer),
                    'inner_fold':int(pair_inner.loc[row.pair_id]),
                    'sample_id':row.sample_id,
                    'pair_id':row.pair_id,
                    'model':'M1',
                    'condition':'A',
                    'score_role':'training_inner_oof_human',
                    'calibrated_score':float(inner_prob[position]),
                })
        tf=TurkishAIDetectionFeatureTransformer().fit(tr); Xtr=tf.transform(tr)[feats]; Xte=tf.transform(te)[feats]; mdl=make_logreg(best).fit(Xtr,ytr); dec=mdl.decision_function(Xte); prob=platt_prob(cal,dec); pred=(prob>tau).astype(int)
        met=evaluate_binary(yte,prob,pred); met.update({'outer_fold':outer,'model':feature_set,'best_C':best,'threshold':tau}); fold_rows.append(met)
        for i,(_,r) in enumerate(te.iterrows()): pred_rows.append({'sample_id':r.sample_id,'pair_id':r.pair_id,'label':r.label,'model_generator':r.model,'field':r.field,'broad_field':r.broad_field,'outer_fold':outer,'system':feature_set,'decision_score':float(dec[i]),'prob':float(prob[i]),'pred':int(pred[i]),'threshold':tau})
    result=(pd.DataFrame(pred_rows),pd.DataFrame(fold_rows),pd.DataFrame(tune))
    return (*result,pd.DataFrame(m05_rows)) if collect_m05 else result

def nested_cv_surface(raw,folds,c_grid=(.01,.03,.1,.3,1,3,10),k_exceed=15):
    pred=[]; fm=[]; tune=[]
    for outer in range(5):
        trpairs=folds.loc[folds.outer_fold.ne(outer),'pair_id']; tepairs=folds.loc[folds.outer_fold.eq(outer),'pair_id']; innercol=f'inner_fold_outer{outer}'
        tr=raw[raw.pair_id.isin(trpairs)].copy(); te=raw[raw.pair_id.isin(tepairs)].copy(); ytr=y_from_label(tr.label); yte=y_from_label(te.label)
        Xall=surface_model_matrix(tr)
        crows=[]
        for C in c_grid:
            oof=np.full(len(tr),np.nan)
            for inner in range(4):
                vp=folds.loc[(folds.outer_fold.ne(outer))&folds[innercol].eq(inner),'pair_id']; mask=tr.pair_id.isin(vp).to_numpy(); mdl=make_logreg(C).fit(Xall.iloc[np.where(~mask)[0]],ytr[~mask]); oof[mask]=mdl.decision_function(Xall.iloc[np.where(mask)[0]])
            crows.append({'outer_fold':outer,'model':'M3','C':float(C),'std_pauc':standardized_pauc(ytr,oof)})
        tune.extend(crows); best=_best_c(crows); innerdec=np.full(len(tr),np.nan)
        for inner in range(4):
            vp=folds.loc[(folds.outer_fold.ne(outer))&folds[innercol].eq(inner),'pair_id']; mask=tr.pair_id.isin(vp).to_numpy(); mdl=make_logreg(best).fit(Xall.iloc[np.where(~mask)[0]],ytr[~mask]); innerdec[mask]=mdl.decision_function(Xall.iloc[np.where(mask)[0]])
        cal=fit_platt(innerdec,ytr); ip=platt_prob(cal,innerdec); tau=order_stat_threshold(ip[ytr==0],k_exceed); mdl=make_logreg(best).fit(Xall,ytr); dec=mdl.decision_function(surface_model_matrix(te)); prob=platt_prob(cal,dec); pr=(prob>tau).astype(int); met=evaluate_binary(yte,prob,pr); met.update({'outer_fold':outer,'model':'M3','best_C':best,'threshold':tau}); fm.append(met)
        for i,(_,r) in enumerate(te.iterrows()): pred.append({'sample_id':r.sample_id,'pair_id':r.pair_id,'label':r.label,'model_generator':r.model,'field':r.field,'broad_field':r.broad_field,'outer_fold':outer,'system':'M3','decision_score':float(dec[i]),'prob':float(prob[i]),'pred':int(pr[i]),'threshold':tau})
    return pd.DataFrame(pred),pd.DataFrame(fm),pd.DataFrame(tune)

def make_tfidf_pipeline(C):
    from sklearn.pipeline import FeatureUnion,Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    feats=FeatureUnion([
      ('word',TfidfVectorizer(lowercase=False,ngram_range=(1,2),min_df=2,sublinear_tf=True,norm='l2')),
      ('char',TfidfVectorizer(lowercase=False,analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True,norm='l2'))])
    return Pipeline([('tfidf',feats),('lr',LogisticRegression(C=float(C),solver='liblinear',max_iter=5000,random_state=42))])

def nested_cv_tfidf(raw,folds,text_col,c_grid=(.01,.03,.1,.3,1,3,10),k_exceed=15):
    pred=[]; fm=[]; tune=[]
    texts=raw[text_col].map(turkish_lower)
    for outer in range(5):
        trpairs=folds.loc[folds.outer_fold.ne(outer),'pair_id']; tepairs=folds.loc[folds.outer_fold.eq(outer),'pair_id']; innercol=f'inner_fold_outer{outer}'
        tr=raw[raw.pair_id.isin(trpairs)].copy(); te=raw[raw.pair_id.isin(tepairs)].copy(); ytr=y_from_label(tr.label); yte=y_from_label(te.label); ttr=tr[text_col].map(turkish_lower); tte=te[text_col].map(turkish_lower)
        crows=[]
        for C in c_grid:
            oof=np.full(len(tr),np.nan)
            for inner in range(4):
                vp=folds.loc[(folds.outer_fold.ne(outer))&folds[innercol].eq(inner),'pair_id']; mask=tr.pair_id.isin(vp).to_numpy(); mdl=make_tfidf_pipeline(C).fit(ttr.iloc[np.where(~mask)[0]],ytr[~mask]); oof[mask]=mdl.decision_function(ttr.iloc[np.where(mask)[0]])
            crows.append({'outer_fold':outer,'model':'M4','C':float(C),'std_pauc':standardized_pauc(ytr,oof)})
        tune.extend(crows); best=_best_c(crows); innerdec=np.full(len(tr),np.nan)
        for inner in range(4):
            vp=folds.loc[(folds.outer_fold.ne(outer))&folds[innercol].eq(inner),'pair_id']; mask=tr.pair_id.isin(vp).to_numpy(); mdl=make_tfidf_pipeline(best).fit(ttr.iloc[np.where(~mask)[0]],ytr[~mask]); innerdec[mask]=mdl.decision_function(ttr.iloc[np.where(mask)[0]])
        cal=fit_platt(innerdec,ytr); ip=platt_prob(cal,innerdec); tau=order_stat_threshold(ip[ytr==0],k_exceed); mdl=make_tfidf_pipeline(best).fit(ttr,ytr); dec=mdl.decision_function(tte); prob=platt_prob(cal,dec); pr=(prob>tau).astype(int); met=evaluate_binary(yte,prob,pr); met.update({'outer_fold':outer,'model':'M4','best_C':best,'threshold':tau}); fm.append(met)
        for i,(_,r) in enumerate(te.iterrows()): pred.append({'sample_id':r.sample_id,'pair_id':r.pair_id,'label':r.label,'model_generator':r.model,'field':r.field,'broad_field':r.broad_field,'outer_fold':outer,'system':'M4','decision_score':float(dec[i]),'prob':float(prob[i]),'pred':int(pr[i]),'threshold':tau})
    return pd.DataFrame(pred),pd.DataFrame(fm),pd.DataFrame(tune)

def stratified_pair_bootstrap(pred_df,folds,B=2000,seed=42):
    """OOF FPR/TPR intervals, resampling pairs inside outer_fold x model x field cells."""
    meta=folds[['pair_id','outer_fold','model','field']].copy(); rng=np.random.default_rng(seed); groups=[g.pair_id.tolist() for _,g in meta.groupby(['outer_fold','model','field'],sort=True)]
    vals=[]
    for _ in range(B):
        sampled=[]
        for ids in groups: sampled.extend(rng.choice(ids,size=len(ids),replace=True).tolist())
        # duplicate pair IDs must preserve multiplicity
        pieces=[pred_df[pred_df.pair_id.eq(pid)] for pid in sampled]; bdf=pd.concat(pieces,ignore_index=True); y=y_from_label(bdf.label); pr=bdf.pred.to_numpy(int)
        h=y==0; a=y==1; vals.append((float(pr[h].mean()),float(pr[a].mean())))
    arr=np.asarray(vals); return {'fpr_ci95':np.quantile(arr[:,0],[.025,.975]).tolist(),'tpr_ci95':np.quantile(arr[:,1],[.025,.975]).tolist()}

def summary_outer(pred,fold_metrics,folds=None,bootstrap_B=2000):
    fpr=float(pred.loc[pred.label.eq('human'),'pred'].mean()); tpr=float(pred.loc[pred.label.eq('ai'),'pred'].mean()); out={'mean_outer_auc':float(fold_metrics.auc.mean()),'sd_outer_auc':float(fold_metrics.auc.std(ddof=1)),'observed_oof_fpr':fpr,'observed_oof_tpr':tpr,'outer_auc':fold_metrics[['outer_fold','auc']].to_dict('records')}
    if folds is not None: out.update(stratified_pair_bootstrap(pred,folds,B=bootstrap_B))
    return out

def assemble_raw_table(base,sentence,sss,mes,tbs):
    meta=base.copy(); keep=['sample_id','pair_id','label','model','field','broad_field','year','text_A','text_B']
    keep=[c for c in keep if c in meta.columns]; out=meta[keep].copy()
    for df in [sentence,sss,mes,tbs]:
        cols=[c for c in df.columns if c=='sample_id' or c not in out.columns]
        out=out.merge(df[cols],on='sample_id',how='left',validate='one_to_one')
    if len(out)!=len(base):raise ValueError('Feature merge changed row count')
    return out
