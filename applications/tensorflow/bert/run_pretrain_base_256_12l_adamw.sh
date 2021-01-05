export GM2=mk2
export POPLAR_LOG_LEVEL=INFO
export TF_LOG_LEVEL=INFO
export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"/localdata/liguoying/profiles/chinese_bert/base_12l_bs16_ee222222_amp0.11", "autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.useUnstableFormat": "true"}'
python run_pretraining.py --config configs/pretrain_base_256_12l_adamw.json