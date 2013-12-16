quicksand
===================

Turning Terra into a probabilistic programming language

Depends on
 - [terra-utils](https://github.com/dritchie/terra-utils)
 - [terra-ad](https://github.com/dritchie/terra-ad)

 You'll need to modify the `LUA_PATH` environment variable to point to these libraries. It's also a good idea to add an entry for quicksand itself. For example (on Linux/OSX), if all my repositories live in ~/code, I might have the following in my shell config file:

```
     export LUA_PATH="?.t;~/code/terra-utils/?.t;~/code/terra-ad/?.t;~/code/quicksand/?.t;~/code/quicksand/?/init.t"
```

 The last entry allows the entire quicksand package to be imported with `terralib.require("prob")`