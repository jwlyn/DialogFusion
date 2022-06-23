
conda env create -f environment.yml
```
mkdir selfchat
parlai self_chat --model-file zoo:blender/blender_1Bdistill/model --inference nucleus --num-self-chats 20 --task blended_skill_talk --include-personas True --include-initial-utterances True --outfile selfchat/merge_sgd_20.json
parlai self_chat --model-file zoo:blender/blender_1Bdistill/model --inference nucleus --num-self-chats 20 --task blended_skill_talk --include-personas True --include-initial-utterances True --outfile selfchat/simulators_20.json
```

