
import spacy
spacy_nlp = spacy.load("en_core_web_sm")

def doc_chunks_creator(doc_text, chunk_len, stride):
    '''Create list of sentence chunks from a big document text.
    Using spacy sentence segmentation to create sentences and 
    taking chunk_len sentences in one chunk
    with a given stride'''

    doc = spacy_nlp(doc_text)
    doc_sents = [sent.text for sent in doc.sents]

    doc_chunks = []
    chunk_start = 0
    while(chunk_start < len(doc_sents)):
        chunk_text = ' '.join(doc_sents[chunk_start:chunk_start+chunk_len])
        doc_chunks.append(chunk_text)
        chunk_start += stride
    
    return doc_chunks





