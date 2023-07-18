import torch
import spacy
import librosa
from moviepy.editor import AudioFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
#from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Download the mp4 file from (https://www.ted.com/talks/nigel_rothfels_can_zoos_actually_save_species_from_extinction)



def video_to_audio(video_path, audio_path):
    my_audio_clip = AudioFileClip(video_path)
    my_audio_clip.write_audiofile(audio_path)


def audio_to_text(audio_path):
    # load model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    segment_length = 60
    all_text = ''

    for i in range(5):
        # read the sound file
        start_at = i * 60
        speech, rate = librosa.load(audio_path, sr=16000, offset=start_at, duration=segment_length)

        # Tokenize the waveform
        
        input_values = tokenizer(speech, return_tensors='pt').input_values
        # retrieve logits from the model
        logits = model(input_values).logits
        
        # take argmax value and decode into transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)
        all_text += transcription[0]

    return all_text


def extract_keywords(text, nlp, top_n=5):
    model = KeyBERT(model=nlp)
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=top_n)
    return keywords


def create_word_cloud(keywords):
    keywords_text = " ".join([kw[0] for kw in keywords])
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(keywords_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def create_bar_chart(keywords):
    keywords_list = [kw[0] for kw in keywords]
    frequencies = [kw[1] for kw in keywords]
    x_labels = [kw.replace(' ', '\n') for kw in keywords_list]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots()
    rects = ax.bar(x, frequencies)
    ax.set_ylabel('Frequencies')
    ax.set_title('Keywords Frequencies')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects)
    fig.tight_layout()
    plt.show()


def main():
    video_path = '165004.mp4'
    audio_path = '165004.wav'

    video_to_audio(video_path, audio_path)
    print('Convert Video to Audio End')

    all_text = audio_to_text(audio_path)
    print('\n============')
    print(all_text)
    print('============\n')
    print("\n Speech Recognition End")

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

    print('\n=========recognized text keyword extraction: ============')
    keywords_recognized = extract_keywords(all_text, nlp)
    for kw in keywords_recognized:
        print(kw)

    print('\n=========initial text keyword extraction: ============')
    initial_text = """
    For thousands of years, native Takhi horses roamed the steppes of Central Asia. But by the late 1960s, they’d become extinct in the wild— the last herds struggling in meager habitats against hunters and competition from local livestock. Some small groups of Takhi survived in European zoos, but their extinction still seemed inevitable. To prevent this terrible fate, a coalition of scientists and zoos pulled together to start an international Takhi breeding program. By the 1990s, these collaborators in Europe and the US began releasing new generations of Asia’s ancient wild horse back into their native habitat. This Takhi revival was a world-famous conservation victory, but the full story is much more complicated than it first appears. And its twists and turns raise serious questions about the role of zoos and what conservation even means. To get the whole story, we need to start in the late 1800s when Russian explorer Nikolay Przhevalsky was gifted the remains of one of these wild horses. Though the Takhi had long been known to local Mongolians, European scientists were intrigued by the remains, which looked more like those of a donkey or zebra than any known domestic horse. They concluded the species was a sort of missing link between wild asses and modern horses. And as reports of the newly dubbed Przhevalsky’s horse circulated through Europe and America, zoo proprietors became eager to acquire the previously unknown species. At this time, zoos were focused primarily on drawing visitors with exotic animals, and their exhibitions were more concerned with entertainment than animal welfare. But in the early 1900s, the near extinction of the American bison and the total extinction of other species like the passenger pigeon inspired zoos to rebrand as centers for conservation. And as it became clear that Przhevalsky’s horse might be headed for a similar fate, zoos began breeding programs to sustain the captive population. However, the individuals behind these programs came to an interesting conclusion about how the horses should be bred. Like their colleagues, they believed the species represented a missing link between modern domestic horses and their more primitive ancestors. They also knew that some of the horses in their collections weren’t purebred Takhi, and many didn’t even resemble the species’ standard description. So breeders felt it was up to them to determine what a wild Takhi should look like, and breed them accordingly. Basing their work on just a few specimens and broad beliefs about what a primitive horse might look like, they created a rigorous model for the ideal Takhi. And over the 20th century, breeders in western zoos and private collections created a population of thousands of horses all carefully bred to share the same physical characteristics. Of course, in their native habitat, wild Takhi had regularly interbred with domesticated horses for millennia, producing a population with much more diverse appearances. So when it was time to introduce the Takhi to their ancestral home, they were quite different from the horses who’d been taken from those steppes a century earlier. Complicating things even further, while these new Takhi herds were no longer in zoos, to this day, almost all remain closely monitored and controlled for their own protection. So in a strange way, it’s hard to say if these animals are actually in the wild or even if they’re truly Takhi. The story of the Takhi horse is not unique. In many of our conservation victories, it’s difficult to say exactly what was saved, and the role that zoos play in conservation can be very complicated. It's clear that zoos have been and can continue to be significant forces for animal preservation, especially efforts to save charismatic animals from extinction. But today, the most direct cause of animal extinctions are humanity’s impacts on animal habitats and Earth’s climate. So if zoos truly want to help protect the diversity of animal life on this planet, perhaps they should redirect their efforts to preserving the natural habitats these animals so desperately need.
    """
    keywords_initial = extract_keywords(initial_text, nlp)
    for kw in keywords_initial:
        print(kw)

    print("End")

    keywords_combined = keywords_recognized + keywords_initial
    print("\n========= Word Cloud Visualization ===========")
    create_word_cloud(keywords_combined)

    print("\n========= Bar Chart Visualization ===========")
    create_bar_chart(keywords_combined)    


if __name__ == "__main__":
    main()
