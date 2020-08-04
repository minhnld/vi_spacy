import spacy
from nltk import Tree

nlp = spacy.load('vi_spacy_model')
doc = nlp('Thời gian xe bus B3 từ Đà Nẵng đến Huế ?')
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

# def to_nltk_tree(node):
#     if node.n_lefts + node.n_rights > 0:
#         return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
#     else:
#         return node.orth_
def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
def nltk_spacy_tree(sent):
    """
    Visualize the SpaCy dependency tree with nltk.tree
    """
    doc = nlp(sent)
    def token_format(token):
        return "_".join([token.orth_, token.tag_, token.dep_])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(token_format(node),
                       [to_nltk_tree(child) 
                        for child in node.children]
                   )
        else:
            return token_format(node)

    tree = [to_nltk_tree(sent.root) for sent in doc.sents]
    # The first item in the list is the full tree
    tree[0].draw()

# [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
nltk_spacy_tree('Xe bus nào đến thành phố Huế lúc 20:00HR ?')
nltk_spacy_tree('Thời gian xe bus B3 từ Đà Nẵng đến Huế ?')
nltk_spacy_tree('Xe bus nào đến thành phố Hồ Chí Minh ?')
nltk_spacy_tree('Những xe bus nào đi đến Huế ?.')
nltk_spacy_tree('Những xe nào xuất phát từ thành phố Hồ Chí Minh ?.')
nltk_spacy_tree('Những xe nào đi từ Đà nẵng đến thành phố Hồ Chí Minh ?.')
