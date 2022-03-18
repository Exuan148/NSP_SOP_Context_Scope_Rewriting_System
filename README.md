# NSP_SOP_Context_Scope_Rewriting_System
the NSP/SOP Context-Scope Rewriter: After the extractive summarization, we use the NSP (Next Sentence Prediction) and SOP (Sentence Order Prediction) tasks of the BERT model to reorder the sentence sequence; In the abstractive summarization, we apply the group subtag-based attention mechanism problem to the seq2seq situation. To further reduce redundancy and irrelevance, each extracted sentence is taken as the input of the rewriter altogether, with its context within a specific scope.
