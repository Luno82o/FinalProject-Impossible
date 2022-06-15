"""
https://blog.csdn.net/wangqianqianya/article/details/89605298
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence
 
sound = AudioSegment.from_mp3("movie300.wav")
loudness = sound.dBFS
#print(loudness)
 
chunks = split_on_silence(sound,
    # must be silent for at least half a second,沉默半秒
    min_silence_len=430,
 
    # consider it silent if quieter than -16 dBFS
    silence_thresh=-45,
    keep_silence=400
 
)
print('总分段：', len(chunks))
 
# 放弃长度小于2秒的录音片段
for i in list(range(len(chunks)))[::-1]:
    if len(chunks[i]) <= 2000 or len(chunks[i]) >= 10000:
        chunks.pop(i)
print('取有效分段(大于2s小于10s)：', len(chunks))
 
'''
for x in range(0,int(len(sound)/1000)):
    print(x,sound[x*1000:(x+1)*1000].max_dBFS)
'''
 
for i, chunk in enumerate(chunks):
    chunk.export("cutFilter300/chunk{0}.wav".format(i), format="wav")
    #print(i)