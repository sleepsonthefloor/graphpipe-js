var assert = require('assert')
var axios = require('axios')
var ndarray = require('ndarray')
var flatbuffers = require('flatbuffers').flatbuffers;
var graphpipefb = require('./graphpipe_generated').graphpipe;


function toGPType (dType) {
  switch(dType) {
    case "int8":
      return graphpipefb.Type.Int8
    case "uint8":
      return graphpipefb.Type.Uint8
    case "uint8_clamped":
      return graphpipefb.Type.Uint8
    case "int16":
      return graphpipefb.Type.Int16
    case "uint16":
      return graphpipefb.Type.Uint16
    case "int32":
      return graphpipefb.Type.Int32
    case "uint32":
      return graphpipefb.Type.Uint32
    case "float32":
      return graphpipefb.Type.Float32
    case "int64":
      return graphpipefb.Type.Int64
    case "uint64":
      return graphpipefb.Type.Uint64
    case "float64":
      return graphpipefb.Type.Float64
    default:
      throw "Unknown dtype: " + dType
  }
}

function output (builder, offset) {
  builder.finish(offset)
  var buf = builder.asUint8Array()
  return new Buffer(buf, 'binary')
}


function serializeMetadataRequest () {
  var builder = new flatbuffers.Builder(1024);

  graphpipefb.MetadataRequest.startMetadataRequest(builder)
  var metadataReq = graphpipefb.MetadataRequest.endMetadataRequest(builder)

  graphpipefb.Request.startRequest(builder)
  graphpipefb.Request.addReqType(builder, graphpipefb.Req.MetadataRequest)
  graphpipefb.Request.addReq(builder, metadataReq)

  var reqOffset = graphpipefb.Request.endRequest(builder)
  return output(builder, reqOffset)
}


function deserializeMetadataResponse (data) {
  var buf = new flatbuffers.ByteBuffer(data)
  return graphpipefb.MetadataResponse.getRootAsMetadataResponse(buf)
}


function serializeNDArray (builder, nda) {
  var shape64 = []
  for (var i=0; i < nda.shape.length; i++) {
    var low = nda.shape[i] & 0xFFFFFFFF
    var high = 0
    shape64.push({ low: low, high: high })
  }

  var shapeOffset = graphpipefb.Tensor.createShapeVector(builder, shape64)

  // this is slow :/
  var dataOffset = graphpipefb.Tensor.createDataVector(builder, Buffer(nda.data.buffer, 'binary'))

  graphpipefb.Tensor.startTensor(builder)
  graphpipefb.Tensor.addShape(builder, shapeOffset)
  graphpipefb.Tensor.addType(builder, toGPType(nda.dtype))
  graphpipefb.Tensor.addData(builder, dataOffset)
  var tensorOffset = graphpipefb.Tensor.endTensor(builder)
  return tensorOffset
}


function serializeNDArrays (builder, ndas) {
  var rval = []
  for (var i=0; i<ndas.length; i++) {
    rval.push(serializeNDArray(builder, ndas[i]))
  }
  
  return rval
}


function serializeInferRequest (ndarrays, inputNames, outputNames, config) {
  if (!Array.isArray(outputNames)) {
    ndarrays = [ndarrays]
  }

  var builder = new flatbuffers.Builder(1)
  var tensors = serializeNDArrays(builder, ndarrays)
  var tensorsVectorOffset = graphpipefb.InferRequest.createInputTensorsVector(builder, tensors)

  config = config || ""
  var configOffset = builder.createString(config)

  inputNames = inputNames || []
  if (inputNames) {
    if (!Array.isArray(inputNames)) {
      inputNames = [inputNames]
    }
  }
  var inputNamesOffset = graphpipefb.InferRequest.createInputNamesVector(builder, inputNames)

  outputNames = outputNames || []
  if (outputNames) {
    if (!Array.isArray(outputNames)) {
      outputNames = [outputNames]
    }
  }
  var outputNamesOffset = graphpipefb.InferRequest.createOutputNamesVector(builder, outputNames)

  graphpipefb.InferRequest.startInferRequest(builder)
  graphpipefb.InferRequest.addInputTensors(builder, tensorsVectorOffset)
  graphpipefb.InferRequest.addInputNames(builder, inputNamesOffset)
  graphpipefb.InferRequest.addOutputNames(builder, outputNamesOffset)
  graphpipefb.InferRequest.addConfig(builder, configOffset)

  var inferReqOffset = graphpipefb.InferRequest.endInferRequest(builder)

  graphpipefb.Request.startRequest(builder)
  graphpipefb.Request.addReqType(builder, graphpipefb.Req.InferRequest)
  graphpipefb.Request.addReq(builder, inferReqOffset)
  var reqOffset = graphpipefb.Request.endRequest(builder)
  return output(builder, reqOffset)
}


function tensorToTypedArray (outputTensor) {
  var tmp = (outputTensor.dataArray())
  var buf = tmp.buffer.slice(tmp.byteOffset, tmp.byteOffset + tmp.byteLength)

  switch(outputTensor.type()) {
    case graphpipefb.Type.Int8:
      return new Int8Array(buf)
    case graphpipefb.Type.Uint8:
      return new Uint8Array(buf)
    case graphpipefb.Type.Uint16:
      return new Uint16Array(buf)
    case graphpipefb.Type.Int16:
      return new Int16Array(buf)
    case graphpipefb.Type.Int32:
      return new Int32Array(buf)
    case graphpipefb.Type.Uint32:
      return new Uint32Array(buf)
    case graphpipefb.Type.Float32:
      return new Float32Array(buf)
    case graphpipefb.Type.Float64:
      return new Float64Array(buf)
    case graphpipefb.Type.Int64:
    case graphpipefb.Type.Uint64:
    default:
      throw "Unhandled data type"
  }
}

function remote (url, ndarrays, inputNames, outputNames, config, params) {
  function parseInferResponse(resp, headers) {
    var buf = new flatbuffers.ByteBuffer(Buffer(resp))
    var response = graphpipefb.InferResponse.getRootAsInferResponse(buf)
    var rval = []
    for (var i=0; i<response.outputTensorsLength(); i++) {
      var outputTensor = response.outputTensors(i);
      var shape = []
      for (var j=0; j<outputTensor.shapeLength(); j++) {
        var s = outputTensor.shape(j)
        shape.push((s.hi << 32) + s.low)
      }
      var arr = tensorToTypedArray(outputTensor)
      rval.push(ndarray(arr, shape))
    }
    return rval
  }
  var serializedReq = serializeInferRequest(ndarrays, inputNames, outputNames, config)
  if (!params) {
    params = {}
  }
  return axios.request(Object.assign({}, {
          responseType: 'arraybuffer',
          url: url,
          method: 'POST',
          transformResponse: [parseInferResponse],  
          data: serializedReq
      }, params))
}

function parseIOMeta (io) {
  var input = {}
  input['Name'] = io.name()
  input['Description'] = io.description()
  input['Type'] = io.type()
  var shape = []
  for (var j=0; j<io.shapeLength(); j++) {
    var s = io.shape(j)
    shape.push((s.hi << 32) + s.low)
  }
  input['Shape'] = shape
  return input
}

function metadatafb (url) {
  function parseMetadataResponse(resp, headers) {
    var buf = new flatbuffers.ByteBuffer(resp)
    var meta = graphpipefb.MetadataResponse.getRootAsMetadataResponse(buf)
    var rval = {}
    rval['Name'] = meta.name()
    rval['Version'] = meta.version()
    rval['Server'] = meta.server()
    rval['Description'] = meta.description()
    var inputs = []
    var outputs = []
    for (var i=0; i<meta.inputsLength(); i++) {
      var input = parseIOMeta(meta.inputs(i, new graphpipefb.IOMetadata()))
      inputs.push(input)
    }
    for (var i=0; i<meta.outputsLength(); i++) {
      var output = parseIOMeta(meta.outputs(i, new graphpipefb.IOMetadata()))
      outputs.push(output)
    }
    rval['Inputs'] = inputs
    rval['Outputs'] = outputs
    for (var i=0; i<meta.outputsLength(); i++) {
      var inp = meta.outputs(i, new graphpipefb.IOMetadata())
    }
    return rval
  }
  var serializedReq = serializeMetadataRequest()
  return axios.request({
          responseType: 'arraybuffer',
          url: url,
          method: 'POST',
          transformResponse: [parseMetadataResponse],  
          data: serializedReq
      })
}

var Graphpipe = function() {}
Graphpipe.prototype.remote = remote
Graphpipe.prototype.metadata = metadatafb
module.exports = new Graphpipe()

