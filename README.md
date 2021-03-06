# cheleary

Cheleary is a toolkit to build an easy training environment. It implements
different kinds of encodings and network structures based on `keras` and
`tensorflow`. The main focus are learning tasks around `CHEBI` - an
ontology about chemicals
  
## Usage

```
Usage: cheleary [OPTIONS] COMMAND [ARGS]...


Options:
  --help  Show this message and exit.

Commands:
  collect-dl-data  Command line interface for...
  continue         Load existing task and continue training.
  list-encoders    List all available encoders
  list-models      List all available models
  test             Load existing task and run tests with cached test data.
  train            Construct and train a new task.
```
