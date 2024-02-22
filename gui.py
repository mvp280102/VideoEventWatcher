import json
import requests
import streamlit as st

from fastapi import status
from os.path import join, splitext

from ffmpeg import FFmpeg

from ultralytics import YOLO

from constants import NEW_OBJECT, LINE_INTERSECTION


st.set_page_config(layout='wide')


if 'ckpt_root' not in st.session_state:
    st.session_state.ckpt_root = '/home/mvp280102/models/yolov8'

if 'detector_name' not in st.session_state:
    st.session_state.detector_name = None

if 'classes' not in st.session_state:
    st.session_state.classes = {}


def set_classes():
    ckpt_root = st.session_state.ckpt_root
    detector_name = st.session_state.detector_name

    if ckpt_root and detector_name:
        ckpt_path = str(join(ckpt_root, detector_name.lower() + '.pt'))
        st.session_state.classes = {value: key for key, value in dict(YOLO(ckpt_path).names).items()}


def main():
    st.title(body="VideoEventWatcher")

    st.markdown(body="#### Video file:")
    video_object = st.file_uploader(label="Video file:", label_visibility='collapsed', type=['avi'])
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["#### Watching", "#### Performance", "#### Directories",
                                            "#### Credentials", "#### Other"])

    with tab1:
        column1, column2, column3 = st.columns(3)

        with column1:
            target_classes = st.multiselect(label="Target classes:", options=st.session_state.classes)
            target_events = st.multiselect(label="Target events:", options=[NEW_OBJECT, LINE_INTERSECTION])

        with column2:
            line_angle = st.number_input(label="Line angle (from -180 to 180 degrees):",
                                         value=None, min_value=-180, max_value=180)
            column21, column22 = st.columns(2)

            with column21:
                line_point_x = st.number_input(label="Line point X:", value=None)

            with column22:
                line_point_y = st.number_input(label="Line point Y:", value=None)

        with column3:
            sec_before = st.number_input(label="Time before object appearance (sec):",
                                         value=0, min_value=0, max_value=5)
            sec_after = st.number_input(label="Time after object disappearance (sec):",
                                        value=0, min_value=0, max_value=5)

    with tab2:
        column1, column2, column3, column4 = st.columns(4)

        with column1:
            detector_name = st.selectbox(label="Detection model:", key='detector_name', on_change=set_classes,
                                         options=['YOLOv8-n', 'YOLOv8-s', 'YOLOv8-m', 'YOLOv8-l', 'YOLOv8-x'])
            frames_skip = st.number_input(label="Frames skip:", value=0, min_value=0, max_value=10)

        with column2:
            input_size = st.number_input(label="Input image size:",
                                         value=480, min_value=320, max_value=640, step=10)
            new_track_thresh = st.number_input(label="New track threshold:",
                                               value=0.6, min_value=0.0, max_value=1.0, step=0.05)

        with column3:
            track_buffer = st.number_input(label="Tracker buffer size:",
                                           value=150, min_value=50, max_value=250, step=25)
            match_thresh = st.number_input(label="Tracker matching threshold:",
                                           value=0.8, min_value=0.0, max_value=1.0, step=0.05)

        with column4:
            track_low_thresh = st.number_input(label="Tracker low confidence threshold:",
                                               value=0.1, min_value=0.0, max_value=1.0, step=0.05)
            track_high_thresh = st.number_input(label="Tracker high confidence threshold:",
                                                value=0.5, min_value=0.0, max_value=1.0, step=0.05)

    with tab3:
        column1, column2, column3 = st.columns(3)

        with column1:
            inputs_root = st.text_input(label="Input videos:", value='inputs')
            outputs_root = st.text_input(label="Output videos:", value='outputs')

        with column2:
            ckpt_root = st.text_input(label="Model checkpoints:", key='ckpt_root')
            tracks_root = st.text_input(label="Object tracks:", value='tracks')

        with column3:
            frames_root = st.text_input(label="Video frames:", value='frames')
            events_root = st.text_input(label="Detected events:", value='events')

    with tab4:
        column1, column2, column3 = st.columns(3)

        with column1:
            sql_dialect = st.selectbox(label="SQL dialect:", options=['postgresql'])
            db_driver = st.selectbox(label="Database driver:", options=['psycopg2'])

        with column2:
            db_username = st.text_input(label="Username for database:", type='default', value='videoeventwatcher')
            db_password = st.text_input(label="Password for database:", type='password', value='videoeventwatcher')

        with column3:
            db_name = st.text_input(label="Database name:", type='default', value='videoeventwatcher')
            host_name = st.text_input(label="Host name:", type='default', value='localhost')

    with tab5:
        column1, column2, column3 = st.columns(3)

        with column1:
            duplicate_interval = st.number_input(label="Duplicate event ignore interval (sec):",
                                                 value=0.75, min_value=0.25, max_value=1.25, step=0.05)

        with column2:
            fourcc = st.selectbox(label="FOURCC:", options=['XVID'])

        with column3:
            fps = st.number_input(label="FPS rate:", value=25, min_value=15, max_value=30)

    st.divider()

    if video_object is not None:
        if st.button(label="Watch events", use_container_width=True):
            target_labels = [st.session_state.classes[target_class] for target_class in target_classes]
            detector_name = detector_name.lower()
            line_point = [line_point_x, line_point_y] if line_point_x and line_point_y else None

            config = {
                'watcher': {'target_events': target_events,

                            'duplicate_interval': duplicate_interval,
                            'fps': fps,

                            'inputs_root': inputs_root,
                            'frames_root': frames_root},

                'processor': {'detector_name': detector_name,
                              'input_size': input_size,

                              'tracks_root': tracks_root,
                              'ckpt_root': ckpt_root,

                              'track_buffer': track_buffer,
                              'match_thresh': match_thresh,
                              'new_track_thresh': new_track_thresh,
                              'track_low_thresh': track_low_thresh,
                              'track_high_thresh': track_high_thresh,

                              'frames_skip': frames_skip,
                              'target_labels': target_labels,

                              'line_angle': line_angle,
                              'line_point': line_point},

                'extractor': {'outputs_root': outputs_root,
                              'frames_root': frames_root,
                              'tracks_root': tracks_root,
                              'events_root': events_root,

                              'sec_before': sec_before,
                              'sec_after': sec_after,

                              'fourcc': fourcc,
                              'fps': fps},

                'saver': {'sql_dialect': sql_dialect,
                          'db_driver': db_driver,

                          'db_username': db_username,
                          'db_password': db_password,

                          'host_name': host_name,
                          'db_name': db_name}
            }

            response = requests.post(url='http://127.0.0.1:8000/watch',
                                     json={'config': config, 'filename': video_object.name},
                                     timeout=900)

            if response.status_code == status.HTTP_200_OK:
                st.markdown(body="#### Events:")

                column1, column2 = st.columns(2)

                events = json.loads(response.content)

                for index, event in enumerate(events):
                    column = column2 if index % 2 else column1

                    frame_index = event['frame_index']
                    track_id = event['track_id']
                    event_name = event['event_name']

                    event_description = ('Frame - {}, track ID - {}, event name - {}.'
                                         .format(frame_index, track_id, event_name))

                    raw_event_path = str(join(events_root, splitext(video_object.name)[0], '{}-{}.avi'
                                              .format(frame_index, track_id)))
                    converted_event_path = raw_event_path.replace('.avi', '.mp4')

                    ffmpeg = FFmpeg().option('y').input(raw_event_path).output(converted_event_path, vcodec='libx264')
                    ffmpeg.execute()

                    with column:
                        with st.expander(label=event_description):
                            with open(converted_event_path, 'rb') as event_object:
                                st.video(event_object.read())

        st.divider()


if __name__ == '__main__':
    main()
