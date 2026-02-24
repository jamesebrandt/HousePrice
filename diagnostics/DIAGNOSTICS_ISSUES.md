# Model diagnostics — underlying issues

This note explains common causes of odd or misleading behavior in the **Price vs Square Feet [D]** and **Price vs ZIP Median Income [E]** partial-dependence panels.

---

## [D] Price vs Square Feet

**What the plot shows:** For each value of sqft on the x-axis, the y-axis is the model's average predicted price when *only* sqft is varied and all other features are held at their **median** (or grid default).

**Underlying issues:**

1. **"Holding everything else at median" is misleading for sqft.**  
   Square footage is right-skewed and strongly correlated with bedrooms, lot size, and location. The "median" row is a hypothetical home that may not exist in your data. So the curve can look too flat, too steep, or non-linear in ways that don't match real neighborhoods.

2. **Location absorbs most of the signal.**  
   City and ZIP dummies capture a large share of price variation. Once location is fixed at "median city" and "median ZIP," the *marginal* effect of sqft can look small or odd, even though sqft is important in reality.

3. **Outliers in sqft.**  
   A few very large or very small homes stretch the grid. Partial dependence then reflects behavior at extremes where data is thin, which can distort the curve at the ends.

**Practical takeaway:** Treat [D] as "effect of sqft in a synthetic median home," not as the true $/sqft in any single market. For real $/sqft by area, look at actual vs predicted by city (e.g. [G] and [H]).

---

## [E] Price vs ZIP Median Income

**What the plot shows:** Same idea: x-axis = ZIP median household income (Census ACS), y-axis = average predicted price when only that feature is varied and everything else is held at median.

**Underlying issues:**

1. **Lots of zeros.**  
   If Census data is missing for many ZIPs, `feat_median_income` is 0 for those rows. The partial-dependence curve is then driven only by the subset of listings with non-zero income. If that subset is small or unrepresentative, the curve can look flat, jumpy, or wrong.

2. **Collinearity with city/ZIP.**  
   Income is highly correlated with city and ZIP dummies. The model often attributes most "wealthy area" premium to the location dummies, so the *marginal* effect of `feat_median_income` in the partial-dependence plot can look small or noisy even when income matters in reality.

3. **Outlier cities/ZIPs.**  
   Small or atypical places (e.g. Fountain Green, Benjamin) with few ZIPs or unusual income levels act as leverage points. They can pull the curve in one direction or create odd flat segments. Excluding these cities from training (see `exclude_cities` in config) reduces this distortion.

**Practical takeaway:** [E] shows how the model uses income *given* location dummies. A flat or odd curve often means "location dummies already captured this," not that income is useless. Check feature importance [C] and city-level bias [H] to see if income is still contributing.
