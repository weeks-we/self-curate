"""Use existing M1 payload alignment and identical response masks for SFT/signals."""
from self_curate.contracts import TokenizerRef,digest
from .rendering import ChatTemplateTokenizer


class NonThinkingTokenizer:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    def __getattr__(self,name):
        return getattr(self.tokenizer,name)
    def __call__(self,*args,**kwargs):
        return self.tokenizer(*args,**kwargs)
    def apply_chat_template(self,*args,**kwargs):
        # Local mount names need not contain the model family. Extra Jinja kwargs
        # are ignored by templates which do not expose this option.
        kwargs['enable_thinking']=False
        value=self.tokenizer.apply_chat_template(*args,**kwargs)
        # Transformers 5 defaults to BatchEncoding for tokenize=True; the M1
        # alignment protocol intentionally consumes the flat input_ids list.
        if kwargs.get('tokenize',True) and hasattr(value,'keys'):
            return value['input_ids']
        return value


def renderer(tokenizer):
    cached=getattr(tokenizer,'_ordinary_renderer',None)
    if cached is not None:return cached
    identity=TokenizerRef(tokenizer_id=str(tokenizer.name_or_path),revision='local-content-addressed',
          tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
          template_id='ordinary-nonthinking-payload-v1',
          template_hash=digest({'template':tokenizer.chat_template,'enable_thinking':False}),backend='huggingface')
    cached=ChatTemplateTokenizer(NonThinkingTokenizer(tokenizer),identity)
    tokenizer._ordinary_renderer=cached
    return cached
