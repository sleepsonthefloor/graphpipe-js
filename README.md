# graphpipe-js - an implementation of GraphPipe for javascript
This is an implementation of the GraphPipe protocol for Javascript
clients.

## Usage

First, start up a sample server:

```
docker run -it --rm \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-echo \
      --listen=0.0.0.0:9000
```
This is a graphpipe server that simply echoes back whatever
you pass to it.  Now let's make a call to this server:


```
var ndarray = require('ndarray')
var graphpipe = require('./graphpipe')

var arr = new Float32Array(2 * 3 * 4)
var nda = ndarray(arr, [1, 2, 3, 4])
console.log("about")
graphpipe.remote("http://127.0.0.1:9000", nda, "", "", "{}").then(function (response) {
    console.log(response.data);
}).catch(function(error) {
    console.log(error);
});
```

You can also request metadata from the server like so:
```
graphpipe.metadata("http://127.0.0.1:9000").then(function (response) {
    console.log(response.data);
}).catch(function(error) {
    console.log(error);
});
```

## API

**graphpipe.remote(url, ndarrays, inputNames, outputNames, config, axios_options)** : returns an axios request object

* url: url for the GraphPipe server
* ndarrays: one or more ndarrays representing tensor data
* inputNames: an array of inputNames associated with the input ndarrays.  If not specified the defaults specified in the server are expected.
* outputNames: an array of outputNames you want from the remote server.  If not
  specified defaults are used
* config: a string containing configuration data for the remote server (optional)
* axios_options: options you would like to pass to the underlying axios request

**graphpipe.metadata(url)** : returns metadata about the remote server
* url: url for the GraphPipe server
