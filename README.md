# specialized-attention

Simpler specialized attention implementation - adding position attention and improving content attention.

# Tutorial
- See `Specialized Attention Tutorial.ipynb.ipynb`

# Contributing:
## High Level Notes
- Be very modular (i.e many functions and classes), if it works well there might be more than a single paper coming out of this work. For example we will probably try extending it to use this attention in computer vision, so we want the changes to be simple when we go to 2D positional attention ....

## How
- **Important**: **Many docstrings**. Each "public" function should have at least the docstrings for the arguments. Each "private" (i.e starting with an underscore : \_) function should have at least a one phrase explanation of what it does. Please don't over use private function to not write docstrings: those should only be used when it makes no sense to call them from outside their module / class. For crucial function, also give the return type and shape (ex: `forward` of `nn.Module`'s). I unfortunately haven't written the docstrings for `forward`, but will ask you to do it in yours.
- **Important**: always **specify the type / shapes** of the variable in the docstring. For tensors please also give their shapes.
- **Important**: **PEP8 standards Linter**. Use a linter, once the library is cleaner I'll add some automatic linting test to only accept well formated PR. This is important because it looks cleaner, and personally I use auto-formatter which means that if your code is not well formatted, the next time I push something it will reformat everything and make it seem that a certain commit does a lot more than what it should.
- **Important**: **Naming convention** : `any_variable`, `_any_private_func`, `is_any_boolean`. Please use the boolean one, it is a lot easier to understand code when reading it.
- **Important**: **Make PR**'s : Although I won't have time to code in the next 2 months, I'll do your code reviews.
- **Important**: **Add tests and travis CI**. We really need to add tests, I will probably ask you to do it for your new functions. This is important as it gives some usage examples. I'm sorry that I haven't done that, because I was mostly on dev mode (still I should have !). In your case we already did the dev mode, and have some good results so this is important for you (also you will test on real data so tests are more important). Note that I will not ask you to add tests all the current code (you are welcome to do it, but it would be mean), simply the ones you add, which will make the code reviews easier and the library easier to maintain. 
- Commit messages with [numpy convention](https://docs.scipy.org/doc/numpy-dev/dev/gitwash/development_workflow.html#writing-the-commit-message).
- On the long term we will want to support python >= 2.7 and >= 3.5 (which is the version used on the ILLC servers). I currently don't support python 2.7, and I don't think it's crucial while we are in dev mode (the changes are pretty straight forward so we can them at the end).
- For string formatting use [new style](https://pyformat.info/). I.e `'{} = {}'.format(a, b)` instead of `'%s = %s' % (a, b)`.
- In the best scenario you would open github issue with a few lines explaining what you want to and how for larger changes (ex: self-attention). But this is not crucial for now (they are too many important changes to make before we start doing things in a clean way). SO although I strongly encourage doing this, you don't need to.

I might be pretty annoying for the ones noted with **important**. Sorry for that, but I think it's for the best for everyone.

# To Do:
I've added some *TO DO* in the code, these are normally labeled with short / medium / long term. By short term, I mean that we have to absolutely do it very soon in order to publish. Medium Term means that it would be good to get it done before the paper gets published in a conference, but we don't need it before Arxiving it. Long term means that it's for next paper and we shouldn't do it now (although it's good to keep in mind as it might change the way you refactor the code now). I have also added some notes in the code using *NOTA BENE*.

On the higher level to do list there is:

## Short Term
- Self attention (definitely the most important to publish).

## Medium Term

**Refactoring**:
- Build on fastai (this might be useful to do before going to self attention actually). I don't know much on fastai V1 but it seems pretty good / will be maintained / has already some work in language modeling / has a lot of the "state of the art tricks" (although we don't want too many tricks, it's definitely good to know that the model is a lot better than RNN with SOTA tricks).
- Be key / Value / position specific (partly done in this repo) . Until now I had to make it work with Machine. We don't need that anymore. As an example the `encoder` currently returns `(keys, values), hidden, additional` when it should really return `keys, values, hidden, additional`. I was using this as hack while developing, but we now want the library to use key / value / positioning by default.
- clean the use of "Additional" (started in this repo). This is a dictionary I am currently using in the code and "carrying" everywhere, I basically populate it differently depending on the current parameters. This was important during dev mode as I needed a way where I could have a single function whith all the dozens of parameters I was trying (even though some parameters required special inputs / outputs). It was also the simplest way to build on top of `machine` without changing the whole code. Now that we reduced drastically the number of parameters we should definitely define all the inputs / outputs without some hidden dictionary.
- Use Pytorch V1
- make it work with python 2
- test + make it work when resume training ( I didn't have to use this as I was working with smaller datasets, but you will probably have to, in this case you should do this in short term as you will need it to run experiments).
- add tests
- add the output size of tensors in the docstrings
- add docstrings to all of the `forward` methods of any nn.Module. Should write `args` and `return` for those.

- should use layer normalization (but there are currently no easy way to use it on RNN in pytorch without making it a lot slower)

## Longer term
- Make the positioning attention work with images (i.e 2d gaussian)

