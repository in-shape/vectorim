# Vectorim

A simple, no BS vector database for experimentation and prototyping.
This Vector "DB" is not designed for performance or scale, and will probably choke your machine under more than a few thousand entries.
It is to be used in scripting and experimenting with new ideas, without the hassle of actually using a database. It just keeps the vectors next to your data in a class, and pickles it on each update (if enabled). This way the db can be saved across runs avoiding re-embeddding.
Usage:

```python
from vectorim import Vectorim

# create a database
vector_db = Vectorim([[1.0, 0.4, ...], ...], ["Vectorim", "Is", "The", "Best", "Database", ...], file_path="my_db.pkl")

# search!
scores, data = vector_db.search([0.69, 0.42, ...], top_k=3)

# that's it really
```

For the sake of laziness, vectors/matrices can be either lists/nested-lists of floats or numpy arrays of corresponding dimensions. OpenAI API returns a list of floats as of now, but working with numpy is nicer so internally everything is converted to that format. Shouldn't really bother you as it's hidden, just use any one of them.
