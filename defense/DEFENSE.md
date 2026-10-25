# SmartCross — viva questions, answered from the numbers

Every figure below comes from `defense/defense_eval.py` (the served pipeline),
`defense/cf_experiment.py` and `defense/improve_experiment.py` (how choices were
made), on the same split: trained on 2015-02 → 2016-02 (459,137 events),
settings chosen on 2016-02, tested once on 2016-03 → 2016-05 (104,167 events
the model never saw). Confidence intervals are 95%, from 1,000 bootstrap
resamples. The console's Performance view shows the same tables live.

---

## The 30-second version

> We recommend the next banking product for a customer. Two engines: a
> collaborative engine that asks what customers with this exact portfolio added
> next, and a gradient-boosting content engine that reads the customer's
> profile, three months of their holdings history, *and* the collaborative
> engine's scores. Then the economy tilts the ranking and hard eligibility rules
> remove what the customer cannot have. End to end, the right product is in our
> top three **85.3%** of the time, against about **14%** for three random picks
> and **70.5%** for the popularity list, measured on 104,167 customers the model
> never saw.

---

## About the results

**Q1. What is your headline number, and why that one?**
85.3% top-3 for the whole pipeline (CI 85.1–85.6%). Not the model's 90.4%: the
rules remove the right answer for 10.8% of customers, and a customer at the
counter only sees what survives the rules. Quoting 90.4% would overstate it.

**Q2. Isn't that just because a few products are very common?**
That is why popularity is the baseline. Ranking every customer the same way by
popularity reaches 70.5%. The model is +20.0 points above it (CI excludes
zero), and macro F1 is 0.355 against popularity's 0.062 — popularity only ever
recommends the same handful of products.

**Q3. Your macro F1 is 0.355. Isn't that low?**
It is 7.8× the chance level of 0.0455 across 22 products, several of which
appear fewer than 20 times in three months. Over the 13 products with at least a
0.5% base rate it is 0.434. Rare products (mortgage: 10 test events) cannot be
learned at this sample size, and we say so rather than drop them.

**Q4. How do you know the results aren't luck of the split?**
Bootstrap intervals on every metric; each test month scored on its own (90.8%,
89.5%, 90.9% top-3); and a strictly time-based split, so the model never sees a
customer's future.

---

## About the holdings history — the newest change

**Q5. What are the history features?**
48 of them. For each of the 24 products: was it held one observation earlier
(so a product dropped last month is visible), and for how many months in a row
has it been held (capped at 15). They are computed from state *before* the
month being predicted, exactly like the other features.

**Q6. What did they add?**
Against the same model without them: top-1 58.9% → 63.5% (+4.6 pts), top-3
89.0% → 90.4% (+1.46 pts), macro F1 0.277 → 0.355 (+28%). All three intervals
exclude zero. On its own, history is the strongest input group: 88.3% top-3,
macro F1 0.310 — more than holdings, demographics or peer scores alone.

**Q7. Why does history help so much?**
Last month's holdings say *what* a customer has; history says *how they got
there*. A payroll account held for a year marks a settled salaried customer; a
product dropped last month changes what comes next. This is where the winning
Santander Kaggle solutions found most of their signal too.

**Q8. Did adding them change the data? Could they leak?**
No. The event table was rebuilt and the 563,310 events, their labels and the
original 35 features are byte-for-byte identical to before; 48 columns were
added beside them. Every history value comes from months before the event.

**Q9. A hand-typed customer has no history. What does the app do?**
It estimates one — the customer held today's products last month too, for as
long as they have been with the bank — and the console says plainly that the
history is estimated. Customers drawn from the data use their real history.

---

## About the hybrid

**Q10. Your first version blended two engines 50/50. What was wrong with it?**
It was *worse* than the content engine alone: −1.44 points top-3, a real loss.
The collaborative half ranked by lift, which divides by base rate — it measures
surprise, not likelihood — so it put rare products first and scored 31.2%
top-3, below the popularity list.

**Q11. How did you fix it?**
The collaborative engine now ranks by probability: what customers with exactly
this portfolio added next, backing off to single-product co-occurrence when the
portfolio is rare (31.2% → 84.8% top-3 on its own). Instead of averaging, its
44 scores are inputs to the content engine (feature augmentation, Burke's
taxonomy), so the model learns per customer how much to trust the peers.

**Q12. Could the collaborative features leak the answer?**
No. For each training month, peer features are built only from earlier months.
Test rows use tables from the training months only — exactly what the API has.

**Q13. Why keep the collaborative engine?**
It works on its own (14.3 points above popularity), the fusion measurably
improved macro F1 before history was added (+0.012, real), and its counts —
read as lift — are the plain-English reasons the adviser reads out.

**Q14. How did you choose settings without cheating?**
Everything was chosen on the validation month (2016-02), with a criterion fixed
in advance: validation top-3, macro F1 breaking ties. The test months were
scored once. We also report a choice that went wrong earlier: a blend weight
chosen by MAP@7 that turned out poor on test, which we did not re-tune.

---

## About the method

**Q15. Why gradient boosting and not deep learning?**
127 tabular inputs. Boosted trees are the strongest standard choice for tabular
data, train in about two minutes, and stay explainable.

**Q16. Did you try anything else?**
Yes, each judged on the validation month and reported:
- class-weighted training: top-3 81.9%, macro F1 0.301 — real losses
- per-product thresholds: validation macro F1 0.343 → 0.392, but −0.7 pts top-3
  (our limit was 0.2)
- a hyperparameter search over five settings: none beat the current ones
- XGBoost on the GPU: top-3 level, 3–4× faster to train, lower macro F1

**Q17. Are the probabilities meaningful?**
Yes. Expected calibration error is 1.0%: when the model says 60%, it is right
about 60% of the time. That matters because the economy tilt multiplies them.

**Q18. How did you score the baseline fairly?**
Products already held are never recommended, for every engine alike — the first
version forgot this for popularity and scored it at 56.9% instead of 70.5%.
Ties count against every engine.

---

## About the layers after the model

**Q19. Does the macro regime layer improve accuracy?**
No, and we don't claim it does. It changes the #1 product for 1.7% of customers
and leaves top-3 unchanged (90.4%). It is there for business relevance; the
regime is classified from nine real price series (all three test months: STRESS).

**Q20. The rules remove the right answer 10.8% of the time. Isn't that bad?**
It is the cost of a hard gate, measured instead of hidden; 0.4% is missing data.
With a bank's real credit policy in place of illustrative thresholds it should
approach zero. Rules remove rather than down-weight, because a constraint a high
score can outvote is not a constraint.

**Q21. Are the explanations trustworthy?**
They come from the lift matrix (training months only), and only rules backed by
at least 30 customers are quoted. A 45× lift from four customers is never shown.

**Q22. Part A doesn't beat its baseline. Why include it?**
Because that is the finding: loan *purpose* is not recoverable from credit
features, which describe capacity, not intent. That diagnosis is why we moved to
Santander, which has the month-by-month history the problem needs.

---

## If asked "what would you do next?"

1. **A real credit policy** in place of illustrative thresholds.
2. **More data for rare products** — mortgages and loans have 10–20 test events.
3. **Longer history and live feedback** from the recommendations actually shown.

## Numbers to remember

| | |
|---|---:|
| End-to-end top-3 | **85.3%** |
| Model top-3 / top-1 | 90.4% / 63.5% |
| Popularity baseline top-3 | 70.5% |
| Chance: top-3 / top-1 and macro F1 | ~13.6% / 4.5% |
| Macro F1 / × chance | 0.355 / 7.8× |
| Macro F1, 13 learnable products | 0.434 |
| History gain over v2 | +4.6 pts top-1, +1.5 pts top-3, +28% macro F1 |
| CF alone: before → after | 31.2% → 84.8% |
| First hybrid vs content | −1.44 pts (real loss) |
| Right product removed by rules | 10.8% |
| Calibration error | 1.0% |
| Test events / months | 104,167 / 2016-03 → 05 |
