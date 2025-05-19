###Fama-French Three Factor Model####

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data from Excel sheets
df = pd.read_excel('d:\\wf.xlsx', sheet_name='Proxies')

# Define the groups based on the median and percentiles
size_breakpoint = df['Size'].median()
bm_30th_percentile = df['BM'].quantile(0.3)
bm_70th_percentile = df['BM'].quantile(0.7)


# Categorize the funds into groups
df['Size Group'] = df['Size'].apply(lambda x: 'Small' if x <= size_breakpoint else 'Big')
df['BM Group'] = df['BM'].apply(lambda x: 'Low' if x <= bm_30th_percentile else 'Medium' if x <= bm_70th_percentile else 'High')

# Initialize DataFrame to store factor values
factor_values = {'SMB': [], 'HML': []}

# Helper function to calculate returns for a given group
def calculate_group_return(group_by_columns, group_values, return_column):
    group_df = df[(df[group_by_columns] == group_values).all(axis=1)]
    return group_df[return_column].mean()

# Iterate through each returns column
for i in range(1, 61):
    column_name = f'Returns_{i}'

    # SMB (Size Factor)
    SMB = (calculate_group_return(['Size Group', 'BM Group'], ['Small', 'High'], column_name) +
              calculate_group_return(['Size Group', 'BM Group'], ['Small', 'Medium'], column_name) +
              calculate_group_return(['Size Group', 'BM Group'], ['Small', 'Low'], column_name)) / 3 - \
             (calculate_group_return(['Size Group', 'BM Group'], ['Big', 'High'], column_name) +
              calculate_group_return(['Size Group', 'BM Group'], ['Big', 'Medium'], column_name) +
              calculate_group_return(['Size Group', 'BM Group'], ['Big', 'Low'], column_name)) / 3

    # HML (Value factor)
    HML = (calculate_group_return(['Size Group', 'BM Group'], ['Small', 'High'], column_name) +
            calculate_group_return(['Size Group', 'BM Group'], ['Big', 'High'], column_name)) / 2 -\
           (calculate_group_return(['Size Group', 'BM Group'], ['Small', 'Low'], column_name) +
            calculate_group_return(['Size Group', 'BM Group'], ['Big', 'Low'], column_name)) / 2


    # Append the factor values to their respective lists
    factor_values['SMB'].append(SMB)
    factor_values['HML'].append(HML)

# Create a DataFrame for SMB, HML, RMW, and CMA values
factor_df = pd.DataFrame(factor_values)
MKT = pd.read_excel('d:\\wf.xlsx', sheet_name='rf', usecols=['MKT'])
factor_df['MKT'] =MKT['MKT']

# Specify the actual column names instead of using a placeholder


# Load the returns from the Excel file
df = pd.read_excel('d:\\wf.xlsx', sheet_name='Proxies')
returns = df[
    ['Returns_1', 'Returns_2', 'Returns_3', 'Returns_4', 'Returns_5',
     'Returns_6', 'Returns_7', 'Returns_8', 'Returns_9', 'Returns_10',
     'Returns_11', 'Returns_12', 'Returns_13', 'Returns_14', 'Returns_15',
     'Returns_16', 'Returns_17', 'Returns_18', 'Returns_19', 'Returns_20',
     'Returns_21', 'Returns_22', 'Returns_23', 'Returns_24', 'Returns_25',
     'Returns_26', 'Returns_27', 'Returns_28', 'Returns_29', 'Returns_30',
     'Returns_31', 'Returns_32', 'Returns_33', 'Returns_34', 'Returns_35',
     'Returns_36', 'Returns_37', 'Returns_38', 'Returns_39', 'Returns_40',
     'Returns_41', 'Returns_42', 'Returns_43', 'Returns_44', 'Returns_45',
     'Returns_46', 'Returns_47', 'Returns_48', 'Returns_49', 'Returns_50',
     'Returns_51', 'Returns_52', 'Returns_53', 'Returns_54', 'Returns_55',
     'Returns_56', 'Returns_57', 'Returns_58', 'Returns_59', 'Returns_60'
    ]
]

# Load risk-free rates from the same Excel file but different sheet
rf = pd.read_excel('d:\\wf.xlsx', sheet_name='rf')
# Assuming the risk-free rates are in the first column, extract them as a numpy array
rf = rf.iloc[:, 0].values

# Subtract the risk-free rate from each return
excess_returns = returns.sub(rf, axis=1)

# Transpose excess_returns
excess_returns_transposed = excess_returns.T

# Rename columns to TA_1, TA_2, ..., TA_60
excess_returns_transposed.columns = ['TA_' + str(i) for i in range(1, len(excess_returns_transposed.columns) + 1)]

# combine in single df
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M').strftime('%b-%Y')
factor_df.insert(0, 'Date', dates)
combined_df = pd.concat([factor_df.reset_index(drop=True), excess_returns_transposed.reset_index(drop=True)], axis=1)


# Export the combined DataFrame to a single Excel file
combined_df.to_excel('d:\\combined_df.xlsx', index=False)

# Define factors and asset columns
factors = [ 'MKT', 'SMB', 'HML']

# Prepare an empty list to collect regression results
regression_results = []


beta_coefficients = pd.DataFrame(columns=['TA', 'beta_MKT', 'beta_SMB', 'beta_HML'])

# Perform regression for each asset and collect the results
for i in range(1, 1021):  # Assuming Test Assets (TA) from TA_1 to TA_1020
    asset_name = f'TA_{i}'
    Y = combined_df[asset_name]  # Excess return for the asset
    X = combined_df[factors]  # Fama-French factors
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})


    # Create a temporary DataFrame to hold the coefficients for this asset
    temp_df = pd.DataFrame({
        'TA': [asset_name],
        'beta_MKT': [model.params.get('MKT', 0)],
        'beta_SMB': [model.params.get('SMB', 0)],
        'beta_HML': [model.params.get('HML', 0)],
    })

    # Append the temporary DataFrame to the main DataFrame
    beta_coefficients = pd.concat([beta_coefficients, temp_df], ignore_index=True)

# Now you have a DataFrame `beta_coefficients` with all the beta values.
# You can export this DataFrame to an Excel file for later use.
beta_coefficients.to_excel('d:\\beta_coefficients.xlsx', index=False)

# Calculate average excess returns for each asset
average_excess_returns = combined_df.iloc[:, 4:].mean()  # Assuming the excess returns start from the 4th column

# Prepare the independent variables (betas)
X_betas = beta_coefficients[['beta_MKT', 'beta_SMB', 'beta_HML']]

# Adding a constant to the independent variables
X_betas = sm.add_constant(X_betas)

# Prepare the dependent variable (average excess returns)
Y = average_excess_returns
Y = Y.reset_index(drop=True)
X_betas = X_betas.reset_index(drop=True)

# Perform the cross-sectional regression
cross_sectional_model = sm.OLS(Y, X_betas).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

# Coefficients from above regression are risk premium (excluding the constant)
estimated_risk_premiums = cross_sectional_model.params[1:]

# Call the latest value of the risk-free rate (Rf)
rf = pd.read_excel('d:\\wf.xlsx', sheet_name='rf')
Rf = rf.iloc[-1, 0]

# Calculate expected returns using the regression coefficients
expected_returns = pd.DataFrame(beta_coefficients.index, columns=['TA'])
expected_returns.set_index('TA', inplace=True)
expected_returns['Expected_Return'] = Rf + \
    (beta_coefficients['beta_MKT'] * estimated_risk_premiums['beta_MKT']) + \
    (beta_coefficients['beta_SMB'] * estimated_risk_premiums['beta_SMB']) + \
    (beta_coefficients['beta_HML'] * estimated_risk_premiums['beta_HML'])


#Copy the Actual Return into the 'expected_returns' DataFrame
expected_returns['Actual_Return'] = returns.loc[expected_returns.index, 'Returns_60']

#Add a Comparison Column to classify assets
expected_returns['Valuation'] = expected_returns.apply(lambda row: 'undervalued, can be targeted for the investment' if row['Expected_Return'] > row['Actual_Return'] else 'overvalued, must not be targeted for invesmtent', axis=1)

# Add the 'names' column to the destination DataFrame
#expected_returns['Name'] = df['Name']
expected_returns.insert(0, 'Name', df['Name'])
expected_returns.insert(1, 'ISIN', df['ISIN'])
expected_returns.insert(2, 'MF', df['MF'])

# Now, 'expected_returns' DataFrame contains the Expected Return, Actual Return, and Valuation status for each asset
print(expected_returns.head())

#Save this DataFrame to an Excel file
expected_returns.to_excel('d:\\FF3.xlsx', index=False)


import numpy as np
