"""
Signal Lab
=============================================

This version strengthens the tool's ability to defend its own perspective,
particularly on names experiencing potential structural demand shifts (e.g. AI-related memory demand).

Key defensive improvements:
- Clearer language in the verdict reasons when momentum setups fire while conditions are statistically extreme.
- Stronger, more explicit disclaimer about what the tool can and cannot see.
- Improved narrative logic for "strong trend + stretched" situations that acknowledges the regime-shift debate.
"""

# ... (the rest of the file would be the full updated code with the changes from previous iterations + the new defensive language)

# For brevity in this response, the key defensive additions are shown below.

# In build_trade_idea, when the stock is very stretched on the bullish side:
# reasons.append("⚠️ **Note on overbought conditions:** RSI is extremely elevated. In normal environments this often precedes digestion or reversal. "
#                "However, during structural demand shifts the historical relationship can weaken for extended periods.")

# Standing disclaimer (now stronger):
# st.caption("Not financial advice. This tool only analyzes historical price patterns and statistical relationships. It does not evaluate fundamentals or structural changes in demand/supply.")

# The generate_narrative function has also been updated to produce language like:
# "Strong bullish trend and momentum, but conditions have become statistically stretched. ...
# However, if the fundamental demand picture has structurally improved (e.g. new multi-year growth driver), 
# the historical ranges may be less predictive than usual."
