"""Mes Que Un Code -- possession pipeline for More than a Hack 2027, Phase 1.

Modules, in the order data flows through them:

    config.py            every tunable number, with the reason it has the value it has
    loaders.py           ATD, Smart Tagging and homography -> arrays indexed by video frame
    features.py          per-frame evidence: who is near the ball, is the ball dead
    possession_model.py  emission costs + Viterbi decoding over {H, A, D} per half
    postprocess.py       spell clean-up and the frame,label submission writer
    run_solution.py      command line entry point
    experiments.py       baseline-vs-solution comparison table

Everything is indexed by tactical *video* frame (row i of a submission), never by the ATD
frame_id (which is video_frame + 1) or by the hidden IdFrame counter in games.py.
"""
