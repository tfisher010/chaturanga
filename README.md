**Chaturanga**

This library implements policy-free chess engine development using the python-chess library to handle chess-specific considerations, and Tensorflow for model development. 

While game play is handled using the python-chess framework, game states ("boards") are transformed to numpy arrays of shape (8, 8, 12) for model training and scoring. Each vector of the 8x8 grid contains 12 elements - one for each possible piece type (distinguishing between colors; empty spaces are represented by empty arrays). The model itself is a simple CNN with a single convolution and a couple dense layers, trained over a single epoch (validation performance degradation begins immediately when training on random play). I use crossentropy loss and an Adam optimizer, and evaluate performance using simple accuracy.

The resulting engine can beat random play about 85% of the time, and draw another 5-10%. The primary limitation here is the volume of data; I'd like to improve this by writing a loop that performs self-play and model training iteratively, each time using the insights gained from the last round of training. After each self-play stage, data from the previous self-play stage can be discarded to avoid space issues.

While the most sophisticated engines use MCTS for move selection, I'm starting here with a simple minimax strategy. States resulting from possible moves are evaluated solely by the model, without any human heuristics, except that the engine is instructed to checkmate its opponent whenever this is possible on the current move. 

*Example Usage*
```
from engine_development import train_engine
results = train_engine()
```

Author: Ted Fisher
https://www.linkedin.com/in/tfisher010/