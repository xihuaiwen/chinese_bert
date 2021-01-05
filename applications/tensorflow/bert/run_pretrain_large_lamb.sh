export GM2=mk2
export POPLAR_LOG_LEVEL=INFO
export TF_LOG_LEVEL=INFO
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"/localdata/liguoying/profiles/bert-large/bs2_lamb_4ipus", "autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.useUnstableFormat": "true"}'
python run_pretraining.py --config configs/pretrain_large_128_lamb_16ipus.json
