# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sign_data.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fsign_data.proto\x12\x08signData\"\x1e\n\x0eRequestMessage\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\" \n\x0fResponseMessage\x12\r\n\x05reply\x18\x01 \x01(\t2_\n\x11StreamDataService\x12J\n\x13\x62iDirectionalStream\x12\x18.signData.RequestMessage\x1a\x19.signData.ResponseMessageb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sign_data_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REQUESTMESSAGE']._serialized_start=29
  _globals['_REQUESTMESSAGE']._serialized_end=59
  _globals['_RESPONSEMESSAGE']._serialized_start=61
  _globals['_RESPONSEMESSAGE']._serialized_end=93
  _globals['_STREAMDATASERVICE']._serialized_start=95
  _globals['_STREAMDATASERVICE']._serialized_end=190
# @@protoc_insertion_point(module_scope)
