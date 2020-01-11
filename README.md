# Transformers Play Chess

## Introduction

The shannon entropy of natural english language is roughly one byte per word, depending on the dataset used. Shannon estimated the number of possible chess games to be 10^120. I've also seen an estimate of 3 reasonable moves per ply (so 10^40 possible 40 move games). 

This begs the question: just how much information is there in a chess move?

It's a sort of a weird question, seeing as the optimal chess strategy would likely have a (relatively) small number of branches which are able to force a win for white or black, or force a draw. In some sense, measuring the information content of a chess game is a bit like measuring how many mistakes human players make. But I think there is some value in quantifying the number of games which are "interesting to imperfect humans".

### Digression 

Several years ago, I played a card game called "Hanabi". I remember none of the details, except that much of the game revolved sending signals to other players, which might involve making a short-term "suboptimal" move for the sake of transmitting information. (it's very possible i'm misremembering this game horribly). Apparently this game has recently been proposed as a benchmark for multi-agent reinforcement learning.

Somehow inspired by this, I've come up with some increasingly bizzare scenarios involving game play and covertly transmitting information -- suppose you're being held captive, and you're forced to play a chess game which will be broadcast to the world. If you lose you'll be executed, and to win your freedom you need covertly to tell the world where you're being held captive. Can you somehow encode this information in your chess game while not making moves which are too suboptimal?

## Approach

I treated this as a sequence modeling problem. An alternative (and possibly better) approach would be to explicitly make use of the board state. However as I was lazy, I did not do this. I was also motivated by the idea of recreating blindfold chess, which is challenging for humans, but unclear for computers (how would you blindfold a computer? -- (also see Tom Murphy's Elo World)). Also as the "markovian" approach of simply predicting the move given the current board state has been done many, many times before, I decided this was more interesting. 

## Dataset

The lichess.org game database contains at the time of writing roughly 1 billion games.

While I wasn't very picky about getting only the best games, I did want some minimal quality control. Therefore I considered only games where both players had a rating of at least 1510 (I believe new accounts start at a rating of 1500), and where both players had at least 5 minutes to play the game, and where the game was at most 100 moves (200-ply). If both players had a rating of at least 2000, the time requirement was bypassed. Note that for time controls with increment, I converted it into a single number by assuming the game was 40 moves long. Roughly 21 of games passed this first filter. I further divided my dataset up by considering games where both players had a rating of at least 2000 and the time control was at least 5 minutes. Less than 1% of games met this filter, but I didn't find this too worrying as that was still several million games.

Instead of training two different models, or fine-tuning a trained model on the "better" games, I simply added two new tokens to the vocabulary, A, and B, and prefaced each game with one of the two tokens. A was used for the more stringent filter, and B for the rest. I did this primarily to save time. Note that it's fairly trivial to "undo" this conditioning just by summing over the two possible first tokens. I was hoping strategy this would allow me to train with a massive dataset, but then to condition on A to generate higher quality games.

In retrospect, even strong players sometimes run low on time and who wins the game is determined by who moves their pieces faster, with almost total disregard for how much sense the move makes. Since most of the games have each move individually timestamped, it would be ideal to filter out games in which some sort of time crunch happens. However I did not do this.

I chose to use long algebraic notation, which specifies the start and end coordinate of every piece moved (for example, e2e4). "special" moves also include castling and promotion. There are slightly less than 2000 valid unique tokens in this notation. Due to the moderate number of tokens, no BPE was used -- 1-ply == 1 token in the model I trained. I consciously chose not to use short algebraic notation, the standard for humans, because of the amount of implied information and complexity. For example, Ra8 is valid notation for moving a rook to a8, but if more than one rook could possibly move to a8, it must be further disambiguated with something like Rea8 (rook on file e moves to a8). While I strongly suspect the correct choice of notation matters a lot, since I only trained one model I don't have any empirical results along these lines.

Each sequence ends with an additional token to denote which side won, or a draw.

## Model

I used the "transformer_big_single_gpu" (henceforth known as T78) model from the tensor2tensor repository which has roughly 78 million parameters.
I used the default hyperparameters and did not tune anything. I trained on a single 1080ti for almost 4 days (~2 million steps).
This turns out to be roughly 50 million games, which is to say, the model only saw 25% of the dataset.

## Experiments

### Computing the Entropy

I ran the model on 100 "A"-games and 50 "B"-games, corresponding to 7602 and 3690 moves respectively. I did not include the game termination token (draw or win) when computing the entropy. I selected the games randomly from the first week of October 2019, which postdates the entire validation and training data. 

A games: 2.15 bits per ply, 4.43 perplexity

B games: 2.26 bits per ply, 4.80 perplexity

I "preregistered" a guess of 2.5 bits per ply before running any experiments. After seeing the results, I believe a better designed model could probably reach between 1.6 and 2.0 BPP. I also believe a larger model would perform better, as I was probably close to saturating the capacity of T78.

### Sampling

No beam search was used. Invalid moves were masked out. I found that setting the softmax temperature helped produce good quality samples: the temperature was set to 1 for the first 10 ply, 0.25 for the next 10 ply, 0.1 for the next 20 ply, and 0.025 after that. If a repetition is reached, the temperature is bumped up to 0.5, and if another repetition happens, 1.0. These numbers are ad-hoc and no tuning was done. More than 3 repetitions of a position is disabled by masking out the move which would cause repetition. 

Here are 5 non-cherrypicked samples from the model. 

https://lichess.org/MZU59shI
`1. d4 e6 2. c4 Nf6 3. Nc3 Bb4 4. e3 c5 5. Ne2 O-O 6. a3 Bxc3+ 7. Nxc3 cxd4 8. exd4 d5 9. c5 Nc6 10. b4 a6 11. Be2 e5 12. dxe5 Nxe5 13. O-O Be6 14. Bg5 h6 15. Bh4 g5 16. Bg3 Ng6 17. Qd4 Kg7 18. Rad1 Rc8 19. Bf3 Ne7 20. Be5 Nf5 21. Qd2 Kg6 22. g4 Nh4 23. Be2 Ne4 24. Nxe4 dxe4 25. Qe3 Qe7 26. Qxe4+ f5 27. gxf5+ Bxf5 28. Qe3 Rce8 29. f4 gxf4 30. Qxf4 Qg5+ 31. Qxg5+ hxg5 32. Bd6 Rxe2 33. Bxf8 Rg2+ 34. Kh1 Be4 35. Rd6+ Kh5 36. Be7 Nf3 37. Rxf3 Bxf3 38. Rf6 g4 39. Bd6 Re2+ 40. Kg1 Re1+ 41. Kf2 Re2+ 42. Kg3 Rg2+ 43. Kf4 Rxh2 44. Kg3 Rg2+ 45. Kf4 Kh4 46. Rh6# 1-0`

https://lichess.org/xjmfxEmJ
`1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Bg5 Bg7 5. Qd2 O-O 6. O-O-O c6 7. h4 b5 8. f3 Qa5 9. Kb1 b4 10. Nce2 Be6 11. Nc1 Nbd7 12. g4 Nb6 13. h5 Nc4 14. Bxc4 Bxc4 15. hxg6 fxg6 16. Bh6 Rf7 17. Bxg7 Rxg7 18. Nh3 Rb8 19. Ng5 b3 20. cxb3 Bxb3 21. Nxb3 Rxb3 22. axb3 Qxd2 23. Rxd2 h6 24. Ne6 Rf7 25. Rxh6 Nd7 26. Rxg6+ Kh7 27. Rh6+ Kg8 28. Rdh2 Rg7 29. Nxg7 Kxg7 30. Rh7+ Kf6 31. R2h6+ Kg5 32. Rh5+ Kf4 33. Rf5+ Ke3 34. Rxe7 Kxd4 35. Rxd7 c5 36. Rxd6+ Ke3 37. Rxc5 Kxf3 38. g5 Kxe4 39. g6 a5 40. g7 a4 41. g8=Q a3 42. Qd5+ Ke3 43. Rc3+ Kf4 44. Rf6+ Kg4 45. Qf5+ Kh4 46. Rh6# 1-0`

https://lichess.org/1FEKDUfh
`1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. h3 O-O 6. Nf3 e5 7. d5 a5 8. Bg5 Na6 9. Bd3 Nc5 10. Bc2 h6 11. Be3 b6 12. Qd2 Kh7 13. g4 Ng8 14. O-O-O Bd7 15. Rdg1 a4 16. h4 a3 17. b3 Qe7 18. h5 g5 19. Nxg5+ hxg5 20. Bxg5 f6 21. Be3 Bh6 22. g5 fxg5 23. Bxg5 Bxg5 24. Rxg5 Nh6 25. Rhg1 Rg8 26. Rxg8 Rxg8 27. Rxg8 Kxg8 28. Qxh6 Qg7 29. Qxg7+ Kxg7 30. Bd1 Kh6 31. Kd2 Kg5 32. Ke3 Bg4 33. Bxg4 Kxg4 34. h6 Kg5 35. h7 Kg6 36. h8=Q 1-0`

https://lichess.org/p7KdwSt3
`1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bc4 e6 7. Bb3 Be7 8. O-O O-O 9. f4 Nc6 10. Be3 Qc7 11. Qf3 b5 12. a3 Bb7 13. Rad1 Na5 14. Ba2 Nc4 15. Bc1 Rac8 16. Kh1 Rfd8 17. f5 e5 18. Nde2 d5 19. exd5 e4 20. Qh3 Nxd5 21. Nxd5 Bxd5 22. f6 Bxf6 23. Rxf6 gxf6 24. Qg4+ Kh8 25. Nf4 Rg8 26. Qh4 Qe5 27. Nxd5 Rg6 28. Nf4 Rg7 29. b3 Nd6 30. c4 bxc4 31. bxc4 Nf5 32. Qh3 Rcg8 33. Bb2 Qxf4 34. Bxf6 e3 35. Bxg7+ Rxg7 36. Rd8+ Rg8 37. Rxg8+ Kxg8 38. g3 Qe4+ 39. Qg2 Qxg2+ 40. Kxg2 e2 41. Kf2 Nd4 42. c5 Kf8 43. Bc4 a5 44. a4 Ke7 45. Ke3 Nc6 46. Bb5 Nb4 47. c6 Kd6 48. Kd4 f5 49. h4 h5 50. Bxe2 Nxc6+ 51. Kc4 Ne5+ 52. Kb5 Ng4 53. Kxa5 Kc5 54. Ka6 Kb4 55. a5 Nf6 56. Kb6 Nd5+ 57. Kc6 Ne3 58. a6 Nc4 59. a7 Ne5+ 60. Kd5 Ng4 61. a8=Q Ne3+ 62. Ke5 Ng4+ 63. Kxf5 Ne3+ 64. Kg5 Ng4 65. Kxh5 Ne3 66. g4 Ng2 67. g5 Nf4+ 68. Kg4 Nxe2 69. h5 Nd4 70. h6 Ne6 71. h7 Nf8 72. h8=Q Ne6 73. Qf6 Nc5 74. Qd5 Na4 75. Qfd4+ Ka3 76. Qa5 Kb3 77. Qaxa4# 1-0`

https://lichess.org/PTnR6qJD
`1. e4 Nc6 2. d4 e5 3. d5 Nce7 4. c4 d6 5. Nc3 f5 6. Bd3 Nf6 7. Nge2 g6 8. O-O Bg7 9. f4 O-O 10. fxe5 dxe5 11. Bg5 h6 12. Bxf6 Bxf6 13. exf5 gxf5 14. Ng3 e4 15. Bc2 Ng6 16. Qh5 Kg7 17. Nxf5+ Bxf5 18. Rxf5 Bd4+ 19. Kh1 Rxf5 20. Qxf5 Qf6 21. Qxe4 Rf8 22. h3 Be5 23. Re1 Qf4 24. Qxf4 Rxf4 25. b3 Bxc3 26. Re7+ Rf7 27. Rxf7+ Kxf7 28. Bxg6+ Kxg6 29. g4 Kg5 30. Kg2 Kf4 31. Kf2 Ke4 32. Ke2 Kd4 33. h4 Ke4 34. g5 hxg5 35. hxg5 Kf5 36. Kd3 Be5 37. c5 Kxg5 38. Kc4 Kf5 39. b4 Ke4 40. a4 a6 41. b5 axb5+ 42. axb5 Ke3 43. b6 cxb6 44. cxb6 Ke4 45. Kc5 Bd4+ 46. Kd6 Bxb6 47. Ke6 Kd4 48. d6 Bd8 49. d7 b5 50. Kd6 b4 51. Kc6 b3 52. Kd6 b2 53. Ke6 b1=Q 54. Kd6 Qb6# 0-1`

Some observations -- T78 does occasionally leave pieces hanging, but pretty rarely. It can play very well in the opening, decently in the middlegame, but often starts blundering in the endgame. Even in an easily winning position, it can completely forget about a piece and be unable to move it. However given enough time, it usually figures it out.

One of my friends who is a strong amateur chess player played against it and commented that it seems to make the best move more often than most players, but also blunders unnaturally often. More on this below.

### Game Playing

When I saw that [Shawn Presser had trained a similar model](https://slatestarcodex.com/2020/01/06/a-very-unlikely-chess-game/), albeit a much larger one, I decided that I had to pit the two models against each other and let them fight it out.

For game playing, the T78 was conditioned on 'A' as the starting token. I used the same sampling strategy as described above. The GPT2 model sampled greedily (always taking the best move), which should favor it slightly. I noticed Shawn had the clever idea of conditioning it with the header [Result "0-1"] [WhiteElo "2532"] [BlackElo "2763"], (note, GPT2 is always black -- I did not try to mess around with Shawn's code, and it was set up as black by default) to encourage it to produce games in which black wins. I did not use this trick.

https://lichess.org/tqTH8MGS
`1. d3 d5 2. g3 Nf6 3. Nf3 g6 4. Bg2 Bg7 5. O-O O-O 6. Nbd2 c5 7. e4 Nc6 8. Re1 e5 9. c3 d4 10. Nc4 h6 11. a4 Be6 12. Ncxe5 Nxe5 13. Nxe5 Nxe4 14. Rxe4 Bxe5 15. Rxe5 Qd6 16. Bf4 Qxe5 17. Bxe5 Rfe8 18. cxd4 cxd4 19. Bxd4 Rad8 20. Bc3 Rxd3 21. Qxd3 Rd8 22. Qxd8+ Kh7 23. Qf6 Kg8 24. Qg7# 1-0`

https://lichess.org/3EMzKm9V
`1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. c3 d5 5. e5 Qe7 6. cxd4 Bd7 7. Nc3 Nf6 8. Be2 O-O-O 9. O-O Re8 10. Bg5 h6 11. exf6 Qxf6 12. Bxf6 gxf6 13. Nxd5 Bg7 14. Bb5 a6 15. Bxc6 bxc6 16. Nc3 c5 17. dxc5 Be6 18. Qa4 Kb8 19. Qxa6 Rc8 20. Qb5+ Ka8 21. c6 Rb8 22. Qa6# 1-0`

https://lichess.org/A7nTLPwF
`1. c4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Bc5 5. Nb3 Bb6 6. Nc3 Nf6 7. g3 O-O 8. Bg2 Re8 9. O-O d6 10. Bg5 h6 11. Bxf6 Qxf6 12. Nd5 Qd8 13. Nxb6 axb6 14. Nd4 Be6 15. b3 Ra5 16. Qd2 Qd7 17. Rfd1 Rxa2 18. Rxa2 b5 19. Nxe6 fxe6 20. cxb5 e5 21. bxc6 Qxc6 22. Bxc6 bxc6 23. Rc1 e4 24. Rxc6 e3 25. fxe3 Rxe3 26. Rxc7 Rxe2 27. Qxe2 d5 28. Qe8+ Kh7 29. Rxg7+ Kxg7 30. Qe5+ Kg8 31. Qxd5+ Kg7 32. b4 h5 33. b5 h4 34. b6 hxg3 35. hxg3 Kf6 36. b7 Ke7 37. b8=Q Kf6 38. Qb6+ Kg7 39. Qd7+ Kg8 40. Qb8# 1-0`

In the first three games, T78 thoroughly outplays GPT2. It seems that GPT2 is very "capture-happy", often capturing pieces with the queen, only to have its queen captured afterwards. T78 is able to mate GPT2 -- not the most efficiently, but ok enough.

https://lichess.org/1RtevK1T
`1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. Nf3 c5 6. e3 Nc6 7. a3 Bxc3+ 8. Qxc3 b6 9. Be2 Bb7 10. O-O d6 11. Rd1 Qe7 12. b3 e5 13. dxe5 dxe5 14. Bb2 Rad8 15. Rxd8 Rxd8 16. Rd1 Rxd1+ 17. Bxd1 Ne4 18. Qc2 f5 19. Nd2 Nxd2 20. Qxd2 Qd6 21. Qxd6 f4 22. exf4 exf4 23. Bf3 g5 24. Bxc6 Bxc6 25. f3 Kf7 26. Kf2 h5 27. g3 fxg3+ 28. hxg3 h4 29. gxh4 gxh4 30. Be5 h3 31. Kg3 h2 32. Kxh2 Bxf3 33. Kg3 Be2 34. Kf4 Bf3 35. Ke3 Bg2 36. Kf4 Bf3 37. b4 cxb4 38. axb4 a5 39. bxa5 bxa5 40. Ke3 Bg2 41. Kd4 a4 42. c5 a3 43. c6 a2 44. c7 a1=Q+ 45. Kc5 Qa8 46. Kb6 Qb8+ 47. Kc5 Qa8 48. Kb6 Qb8+ 49. Kc5 Qa8 50. Kb5 Qb8+ 51. Ka6 Qa8+ 52. Kb6 `

In this game, GPT2 manages to hold it's ground against T78, despite a T78 having a completely winning position early on. The reason? T78 forgets it has a queen. I stopped this game and declared a draw on move 52 because it wasn't going anywhere.

https://lichess.org/X2EG86ip
`1. e4 e5 2. Ke2 Nc6 3. Kf3 g6 4. g3 Bg7 5. Bg2 d6 6. Ne2 h5 7. h3 h4 8. g4 f5 9. exf5 Bxf5 10. gxf5 gxf5 11. d3 Qd7 12. Nbc3 O-O-O 13. Be3 Nge7 14. Qd2 Kb8 15. Bg5 Rde8 16. Bxe7 Rxe7 17. Nd5 Rf7 18. Ke3 Nd4 19. Nxd4 exd4+ 20. Ke2 c6 21. Nf4 d5 22. Kd1 c5 23. Re1 c4 24. dxc4 dxc4 25. Kc1 b5 26. Kb1 a5 27. a3 a4 28. Qb4 c3 29. bxc3 Qc6 30. Bxc6 Ka7 31. Bxb5 Rb8 32. c4 Rxb5 33. cxb5 Rd7 34. Nd3 Rb7 35. Re7 Rxe7 36. Qxe7+ Ka8 37. Qxg7 Kb8 38. b6 Kc8 39. b7+ Kb8 40. Nc5 Ka7 41. b8=Q+ Kxb8 42. Qb7# 1-0`

In order to give GPT2 some chances, I decided to force T78 to open with the infamous Bongcloud (1. e4 e5 2. Ke2?!). This also has the secondary goal of trying to see how well T78 and GPT2 perform in unknown territory, as this is a very uncommon opening, only played as a joke. Indeed, move 4 is a completely new game in the entire lichess database (by comparison, a T78 "self-play" games often follow previous games up to move 10-15). Nonetheless, T78 manages to pull off a convincing win.

Starting on move 35 of this game, T78 pulls off a nice tactic, trading off the rooks to allow the queen to fork the king and bishop, winning a piece. This is actually not the best move, as there was a mate in 2, and the position is so lopsided it hardly matters. Nonetheless I'm pretty surprised that such a tactic was found.

## Code

See the `src` directory.
