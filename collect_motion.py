"""
collect_motion.py - Guided J & Z motion clip collection
========================================================
Records 30-frame landmark sequences for J and Z.

Usage:
    py collect_motion.py                  # collect both J and Z
    py collect_motion.py --letter J       # only J
    py collect_motion.py --clips 150      # 150 clips per letter
"""

import os, sys, argparse, time
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MOTION_LETTERS, LANDMARK_FEATURES,
    MOTION_CSV_FILE, PROCESSED_DIR,
    MOTION_SEQUENCE_LENGTH, MOTION_SAMPLES_PER_CLASS,
)
from src.landmark_extractor import HandLandmarkExtractor

FONT_S = cv2.FONT_HERSHEY_SIMPLEX
FONT   = cv2.FONT_HERSHEY_DUPLEX
GREEN  = (0, 220, 100)
YELLOW = (0, 220, 220)
RED    = (60,  60, 220)
WHITE  = (240, 240, 240)

MOTION_HINTS = {
    "J": [
        "Start: Pinky finger up (letter I pose)",
        "Trace a J downward then hook LEFT",
        "Keep pinky extended the whole time",
    ],
    "Z": [
        "Start: Index finger pointing RIGHT",
        "Stroke 1: move right  |  Stroke 2: diagonal down-left",
        "Stroke 3: move right again — Z shape done",
    ],
}
MOTION_TIPS = {
    "J": "Do it in ~1 second. Smooth arc, not jerky.",
    "Z": "Three clear strokes. Move your whole arm slightly.",
}


def draw_panel(frame, letter, n_clips, target, state, countdown, trail):
    h, w = frame.shape[:2]
    px = w - 320
    ov = frame.copy()
    cv2.rectangle(ov, (px-10,0),(w,h),(25,25,35),-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.line(frame,(px-10,0),(px-10,h),GREEN,2)

    cv2.putText(frame,"MOTION LETTER",(px,35),FONT_S,0.5,(100,100,110),1,cv2.LINE_AA)
    cv2.putText(frame,letter,(px,115),FONT,3.2,GREEN,5,cv2.LINE_AA)

    cv2.putText(frame,"HOW TO SIGN:",(px,148),FONT_S,0.52,YELLOW,1,cv2.LINE_AA)
    for i,line in enumerate(MOTION_HINTS.get(letter,[])):
        cv2.putText(frame,line,(px,170+i*22),FONT_S,0.48,WHITE,1,cv2.LINE_AA)
    cv2.putText(frame,MOTION_TIPS.get(letter,""),(px,242),FONT_S,0.48,(80,180,80),1,cv2.LINE_AA)

    cv2.putText(frame,f"CLIPS: {n_clips}/{target}",(px,282),FONT_S,0.6,WHITE,1,cv2.LINE_AA)
    bw=280; p=min(n_clips/target,1.0)
    cv2.rectangle(frame,(px,292),(px+bw,308),(50,50,60),-1)
    cv2.rectangle(frame,(px,292),(px+int(bw*p),308),GREEN,-1)

    if state=="WAIT":
        cv2.putText(frame,"SPACE = start clip",(px,348),FONT_S,0.65,YELLOW,2,cv2.LINE_AA)
    elif state=="COUNTDOWN":
        cv2.putText(frame,f"Ready in {countdown}...",(px,348),FONT,1.1,YELLOW,2,cv2.LINE_AA)
    elif state=="RECORDING":
        cv2.circle(frame,(px+10,355),10,RED,-1)
        cv2.putText(frame,"RECORDING!",(px+28,362),FONT_S,0.75,RED,2,cv2.LINE_AA)
        cv2.putText(frame,"Perform motion NOW",(px,388),FONT_S,0.55,WHITE,1,cv2.LINE_AA)

    y=h-100
    for k,d in [("SPACE","Start clip"),("R","Redo last"),("N","Skip letter"),("Q","Save & quit")]:
        cv2.putText(frame,f"[{k}]",(px,y),FONT_S,0.5,GREEN,1,cv2.LINE_AA)
        cv2.putText(frame,d,(px+70,y),FONT_S,0.5,(150,150,160),1,cv2.LINE_AA)
        y+=24

    # Motion trail (index fingertip path)
    for i in range(1,len(trail)):
        alpha=int(220*i/max(len(trail),1))
        cv2.line(frame,trail[i-1],trail[i],(0,alpha,255-alpha),3)

    return frame


def collect_letter(cap, extractor, letter, num_clips, seq_len):
    clips=[]
    state="WAIT"
    countdown=3
    t_start=None
    current=[]
    trail=[]

    while len(clips)<num_clips:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        features,results=extractor.extract_features(frame)
        extractor.draw_landmarks(frame,results,color_override=(0,200,100))

        # Track index fingertip for trail
        if results.multi_hand_landmarks:
            lm=results.multi_hand_landmarks[0].landmark[8]
            fh,fw=frame.shape[:2]
            pt=(int(lm.x*fw),int(lm.y*fh))
            if state=="RECORDING":
                trail.append(pt)
                if len(trail)>seq_len: trail.pop(0)
            else:
                trail=[]

        if state=="COUNTDOWN":
            elapsed=time.time()-t_start
            remaining=3-int(elapsed)
            if remaining<=0:
                state="RECORDING"; current=[]; trail=[]
            else:
                countdown=remaining+1

        elif state=="RECORDING":
            feat=features if features is not None else np.zeros(63,dtype=np.float32)
            current.append(feat)
            if len(current)>=seq_len:
                clips.append(np.array(current[:seq_len],dtype=np.float32))
                print(f"    Clip {len(clips)}/{num_clips} ✓")
                state="WAIT"; trail=[]

        draw_panel(frame,letter,len(clips),num_clips,state,countdown,trail)

        # Top progress bar during recording
        if state=="RECORDING":
            pw=frame.shape[1]-330
            cv2.rectangle(frame,(0,0),(pw,10),(40,40,50),-1)
            cv2.rectangle(frame,(0,0),(int(pw*len(current)/seq_len),10),RED,-1)

        cv2.imshow("Motion Collection",frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord(' ') and state=="WAIT":
            state="COUNTDOWN"; countdown=3; t_start=time.time()
        elif key==ord('r') and state=="WAIT" and clips:
            clips.pop(); print(f"    Redo — now {len(clips)} clips")
        elif key==ord('n'): return []
        elif key==ord('q'): return None

    return clips


def save_clips(clips, letter):
    if not clips: return 0
    os.makedirs(PROCESSED_DIR,exist_ok=True)
    rows=[]
    for ci,clip in enumerate(clips):
        for fi,feat in enumerate(clip):
            r={"gesture":letter,"clip_id":ci,"frame_idx":fi}
            r.update({f"f{i}":feat[i] for i in range(LANDMARK_FEATURES)})
            rows.append(r)
    df=pd.DataFrame(rows)
    mode='a' if os.path.exists(MOTION_CSV_FILE) else 'w'
    df.to_csv(MOTION_CSV_FILE,mode=mode,header=not os.path.exists(MOTION_CSV_FILE),index=False)
    print(f"    {len(clips)} clips saved → {MOTION_CSV_FILE}")
    return len(clips)


def already_collected(letter):
    if not os.path.exists(MOTION_CSV_FILE): return False
    return letter in pd.read_csv(MOTION_CSV_FILE,usecols=["gesture"])["gesture"].values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--letter",default=None,choices=MOTION_LETTERS)
    parser.add_argument("--clips",type=int,default=MOTION_SAMPLES_PER_CLASS)
    parser.add_argument("--redo",action="store_true")
    args=parser.parse_args()

    letters=[args.letter] if args.letter else MOTION_LETTERS
    print(f"\n{'='*55}")
    print(f"  Motion Collection: {' & '.join(letters)}")
    print(f"  {args.clips} clips × {MOTION_SEQUENCE_LENGTH} frames each (~1 sec/clip)")
    print(f"{'='*55}")
    print("  SPACE → 3-sec countdown → perform motion → auto-saved")

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    extractor=HandLandmarkExtractor(max_hands=1)

    for letter in letters:
        if not args.redo and already_collected(letter):
            print(f"\n  [Skip] {letter} already collected (--redo to redo)")
            continue
        print(f"\n  [{letter}] Starting...")
        clips=collect_letter(cap,extractor,letter,args.clips,MOTION_SEQUENCE_LENGTH)
        if clips is None: print("  [Quit]"); break
        elif len(clips)==0: print(f"  [Skipped] {letter}")
        else: save_clips(clips,letter)

    cap.release(); extractor.close(); cv2.destroyAllWindows()

    print(f"\n{'='*55}")
    if os.path.exists(MOTION_CSV_FILE):
        df=pd.read_csv(MOTION_CSV_FILE)
        for ltr,cnt in df.groupby("gesture")["clip_id"].nunique().items():
            print(f"  {ltr}  {'█'*(cnt//5)} {cnt} clips")
    print("\n  Next:  py train_pipeline.py")
    print(f"{'='*55}")

if __name__=="__main__":
    main()