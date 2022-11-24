import os
from nltk.parse.corenlp import CoreNLPServer

STANFORD = os.path.join(".", "stanford-corenlp-4.5.1")

server = CoreNLPServer(
   os.path.join(STANFORD, "stanford-corenlp-4.5.1.jar"),
   os.path.join(STANFORD, "stanford-corenlp-4.5.1-models.jar"),
)

server.start()