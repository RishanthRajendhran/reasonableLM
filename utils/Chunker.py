import spacy
from typing import List, Dict, Any
import logging

SUPPORTED_GRANULARITY_LEVELS = [
    "word",
    "phrase",
    "sentence",
    "paragraph"
]

class Chunker:
    def __init__(self):
        self.chunker = spacy.load("en_core_web_lg")

    def chunk(self, text:str, skip_first_n_tokens:int=0, granularity_level:str="sentence", granularity_len:int=-1)->List[Dict[str, Any]]:
        assert granularity_level in SUPPORTED_GRANULARITY_LEVELS, "Chunking at {} level not supported. Choose from {}".format(granularity_level, SUPPORTED_GRANULARITY_LEVELS)
        if granularity_level not in ["phrase", "paragraph"] and granularity_len != -1:
            logging.warning("granularity_len not used for granularity level {}".format(granularity_level))
        doc = self.chunker(text)
        chunked_text = []
        if granularity_level == "word":
            trailing_whitespace = ""
            for word in doc:
                if word.i < skip_first_n_tokens:
                    trailing_whitespace = word.whitespace_
                    continue
                chunked_text.append({
                    "text": trailing_whitespace + word.text,
                    "start": word.idx-len(trailing_whitespace),
                    "end": word.idx+len(word.text),
                    # "is_punct": word.is_punct,
                    # "is_space": word.is_space,
                    # "is_sent_start": word.is_sent_start,
                    # "is_sent_end": word.is_sent_end,
                })
                trailing_whitespace = word.whitespace_
        elif granularity_level == "sentence":
            for sent in doc.sents:
                if sent.start < skip_first_n_tokens:
                    continue
                chunked_text.append({
                    "text": sent.text,
                    "start": sent.start_char,
                    "end": sent.end_char
                })
        elif granularity_level == "phrase":
            trailing_whitespace = ""
            for sent in doc.sents:
                cur_segment_start, cur_segment_end = -1, -1
                cur_segment = []
                for word in sent:
                    if word.i < skip_first_n_tokens:
                        trailing_whitespace = word.whitespace_
                        continue
                    if (not word.is_punct) and (len(cur_segment) >= granularity_len):
                        chunked_text.append({
                            "text": "".join(cur_segment),
                            "start": cur_segment_start,
                            "end": cur_segment_end,
                        })
                        cur_segment_start = -1
                    if cur_segment_start == -1:
                        cur_segment = []
                        cur_segment_start = word.idx-len(trailing_whitespace)
                    cur_segment.append(trailing_whitespace + word.text)
                    cur_segment_end = word.idx + len(word)
                    trailing_whitespace = word.whitespace_
                if len(cur_segment):
                    chunked_text.append({
                        "text": "".join(cur_segment),
                        "start": cur_segment_start,
                        "end": cur_segment_end,
                    })
        elif granularity_level == "paragraph":
            sents = []
            sent_lens = []
            sent_starts = []
            sent_ends = []
            trailing_whitespace = ""
            for sent in doc.sents:
                if sent.start < skip_first_n_tokens:
                    trailing_whitespace = sent.text_with_ws[len(sent.text):]
                    continue
                sents.append(trailing_whitespace+sent.text)
                sent_lens.append(len(sent))
                sent_starts.append(sent.start_char-len(trailing_whitespace))
                sent_ends.append(sent.end_char)
                trailing_whitespace = sent.text_with_ws[len(sent.text):]

            cur_segment_start, cur_segment_end = -1, -1
            cur_segment = []
            cur_segment_len = 0
            for sent_idx, sent in enumerate(sents):
                if (cur_segment_len >= granularity_len):
                    chunked_text.append({
                        "text": "".join(cur_segment),
                        "start": cur_segment_start,
                        "end": cur_segment_end,
                    })
                    cur_segment_start = -1
                    cur_segment_len = 0
                if cur_segment_start == -1:
                    cur_segment = []
                    cur_segment_len = 0
                    cur_segment_start = sent_starts[sent_idx]
                cur_segment.append(sent)
                cur_segment_end = sent_ends[sent_idx]
                cur_segment_len += sent_lens[sent_idx]
            if len(cur_segment):
                chunked_text.append({
                    "text": "".join(cur_segment),
                    "start": cur_segment_start,
                    "end": cur_segment_end,
                })
        else: 
            raise ValueError("Unrecognized granularity level: {}".format(granularity_level))
        
        return chunked_text