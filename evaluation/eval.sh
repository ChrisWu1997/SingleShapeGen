# LP-IoU and LP-F-score
echo "python eval_LP.py -s $1 -r $2 -g $3";
python eval_LP.py -s $1 -r $2 -g $3
# SSFID
echo "python eval_SSFID.py -s $1 -r $2 -g $3";
python eval_SSFID.py -s $1 -r $2 -g $3
# Diversity (pairwise 1-IoU)
echo "python eval_Div.py -s $1 -g $3";
python eval_Div.py -s $1 -g $3
