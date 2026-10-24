#!/usr/bin/env python3
"""Render a reproducible diagram of the school-isolated modeling workflow."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"reports"/"figures"/"study_workflow.png"
NAVY="#12223B"; TEAL="#137B89"; GOLD="#B58B33"; MUTE="#667486"
BLUE="#E5F0F3"; LIGHT="#EAF5F5"; WARM="#FAF5E9"
fig,ax=plt.subplots(figsize=(12.2,5.15),dpi=190)
fig.patch.set_facecolor("white");ax.set_xlim(0,12.2);ax.set_ylim(0,5.15);ax.axis("off")

def box(x,y,w,h,kicker,body,fill=LIGHT,edge=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.13",
            linewidth=1.25,edgecolor=edge,facecolor=fill,zorder=2))
    ax.text(x+0.14,y+h-0.22,kicker,fontsize=9.8,color=edge,ha="left",va="top",fontweight="bold",zorder=3)
    ax.text(x+0.14,y+h-0.58,body,fontsize=10.3,color=NAVY,ha="left",va="top",
            linespacing=1.24,fontweight="medium",zorder=3)
def arrow(x1,y1,x2,y2,color=MUTE):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
            mutation_scale=13,linewidth=1.7,color=color,zorder=4,shrinkA=3,shrinkB=3))

ax.text(.25,4.96,"STUDY DESIGN  |  FROM DATA AUDIT TO EVALUATED MODEL",fontsize=15.1,
        color=NAVY,fontweight="bold",va="top")
ax.text(.26,4.56,"An institution never appears in both development and locked test data",
        fontsize=10,color=MUTE,va="top")
for a in [
(.25,3.39,2.55,.91,"DATA SOURCE","227,980 rows | 72 columns",BLUE,TEAL),
(3.15,3.39,2.45,.91,"AUDITED CLEANING","212,707 U.S. program rows",BLUE,TEAL),
(5.95,3.39,2.42,.91,"REGRESSION TARGET","25,283 reported bachelor's",BLUE,TEAL),
(8.71,3.39,3.22,.91,"GROUPED SCHOOL SPLIT","By school identifier: opeid6",BLUE,TEAL),
]:box(*a)
for p in [(2.79,3.85,3.16,3.85),(5.59,3.85,5.96,3.85),(8.38,3.85,8.72,3.85)]:arrow(*p)
ax.text(.27,2.97,"DEVELOPMENT BRANCH  |  MODEL SELECTION AND TUNING",fontsize=10.4,fontweight="bold",color=TEAL,va="center")
for a in [
(.25,1.93,2.57,.79,"TRAIN PARTITION","19,512 rows; 1,420 schools",LIGHT,TEAL),
(3.17,1.93,2.61,.79,"GROUPED VALIDATION","5 frozen school-separated folds",LIGHT,TEAL),
(6.15,1.93,2.45,.79,"FOUR MODEL FAMILIES","Ridge | Tree | RF | HGB",LIGHT,TEAL),
(8.95,1.93,2.98,.79,"TUNING AND REFIT","12 training-only HGB settings",LIGHT,TEAL)
]:box(*a)
for p in [(2.83,2.325,3.18,2.325),(5.79,2.325,6.16,2.325),(8.61,2.325,8.96,2.325)]:arrow(*p,color=TEAL)

ax.text(.27,1.56,"TEST BRANCH  |  HOLDOUT EVALUATION AND INTERPRETATION",fontsize=10.4,fontweight="bold",color=GOLD,va="center")
for a in [
(.25,.52,3.05,.79,"LOCKED TEST PARTITION","5,771 rows; 379 unseen schools",WARM,GOLD),
(3.67,.52,3.48,.79,"ONE-TIME FINAL EVALUATION","Tuned MAE $6,957 | R² = 0.779",WARM,GOLD),
(7.50,.52,4.43,.79,"EXPLANATION AND BUSINESS CASE","Feature reliance, group errors, advisor pilot",WARM,GOLD)
]:box(*a)
for p in [(3.31,.92,3.69,.92),(7.16,.92,7.52,.92)]:arrow(*p,color=GOLD)

ax.annotate("Test institutions did not influence feature transforms, model ranking or tuning",
            xy=(.25,.20),fontsize=9.0,color=MUTE,ha="left")
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUTPUT,dpi=190,pad_inches=.13,facecolor="white")
plt.close(fig)
print(OUTPUT, OUTPUT.stat().st_size)
