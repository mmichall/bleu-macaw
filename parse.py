from nltk.parse.corenlp import CoreNLPParser
parser = CoreNLPParser(url='http://localhost:9000')
parse = next(parser.raw_parse("the book I put : in the home box Mustodonts on the table.? John shool"))

print(parse)