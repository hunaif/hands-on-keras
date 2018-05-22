
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np

import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from keras import layers
events_data = '../../data/sample.json'
pos_labels = ['job_added_to_jobcart', 'self_email_job', 'ats_apply_click', 'favorite_job_click']
data = []
MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIMENSION = 8
with open(events_data) as f:
    for line in f:
        row = json.loads(line)
        events = row['collect_list(eventName)']
        if(len(events) > MIN_SEQUENCE_LENGTH ):
            data.append(row['collect_list(eventName)'])
events_data = []
label = []
def label_generator(events_list):
    intersectedList = set(events_list).intersection(set(pos_labels))
    if len(intersectedList) > 0:
        for i,event in enumerate(events_list):
            if(intersectedList.__contains__(event)):
                if(len(events_list[:i]) > MIN_SEQUENCE_LENGTH):
                    events_data.append(events_list[:i])
                    label.append(1)
                    break
    else:
        events_data.append(events_list)
        label.append(0)
# label_generator(["refiner_search","refiner_search","refiner_search","header_menu_click","header_menu_click","ats_apply_click","refiner_search","refiner_search","header_menu_click","job_category_page_view"])
[label_generator(sublist) for sublist in data]
all_events = ["job_click","job_category_page_view","previous_job_click","linkedin_server_event_data","content_modal_next_click","header_menu_click","linkedin_recommendation_dot_slider_click","linkedin_recommended_job_click","job_category_click","sort_by_click","null","clear_searches_click","results_search","Share_Twitter","linkedin_recommendation_previous_slider_click","recommendation_dot_slider_click","content_modal_previous_click","content_click","linkedin_recommended_category_click","content_modal_read_more_click","home_page_view","search_result_multilocation_click","recommendation_next_slider_click","linkedin_logout_click","similar_job_click","refiner_search","recently_viewed_job_click","job_details_view","back_to_search_results_click","Share_LinkedIn","next_job_click","linkedin_recommendation_next_slider_click","Share_Facebook","recommendation_previous_slider_click","linkedin_login_click","search_result_page_view","recommended_job_click","footer_menu_click","landing_page_view","linkedin_profile_data","content_modal_close_click","type_ahead_search","page_search","Share_Googleplus","null","header_category_click"]
VOCAB_SIZE = len(all_events) + 1
events_dict = {}
for i,x in enumerate(all_events):
    events_dict[x] = i+1
def vect_generator(eventslist):
    x = [events_dict[eve] if events_dict.__contains__(eve) else 0 for eve in eventslist]
    return np.array(x)
events_vectors = [vect_generator(x) for x in events_data]
events_vectors = [np.asarray(x) for x in events_vectors]
padded_data = pad_sequences(events_vectors,maxlen=MAX_SEQUENCE_LENGTH)
label = np.asarray(label)
indicies = np.arange(padded_data.shape[0])
np.random.shuffle(indicies)
events_vectors = padded_data[indicies]
label = label[indicies]
X_train, X_test, y_train, y_test = train_test_split(events_vectors, label, test_size=0.2, random_state=1)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
model = Sequential()
model.add(layers.Embedding(VOCAB_SIZE,EMBEDDING_DIMENSION))
model.add(layers.LSTM(16))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['acc'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=8,validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()