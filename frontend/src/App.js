import React, { useState } from 'react';
import GodownSelector from './components/GodownSelector';
import CustomerSelector from './components/CustomerSelector';
import Recommendations from './components/Recommendations';
import { ProductNameProvider } from './components/ProductNameContext';
import './App.css';

function App() {
  const [selectedGodown, setSelectedGodown] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState('');

  // Handler for godown selection
  const handleGodownSelect = (godownCode) => {
    setSelectedGodown(godownCode);
    setSelectedCustomer(''); // Reset customer selection when godown changes
  };

  // Handler for customer selection
  const handleCustomerSelect = (customerId) => {
    setSelectedCustomer(customerId);
  };

  return (
    <ProductNameProvider>
      <div className="app">
        <header className="header">
          <h1>Supermarket Recommendation System</h1>
        </header>
        
        <main className="container">
          <div className="selectors-row">
            <div className="selector-wrapper">
              <GodownSelector 
                selectedGodown={selectedGodown}
                onGodownSelect={handleGodownSelect}
              />
            </div>
            <div className="selector-wrapper">
              <CustomerSelector 
                godownCode={selectedGodown}
                selectedCustomer={selectedCustomer}
                onCustomerSelect={handleCustomerSelect}
              />
            </div>
          </div>
          
          <div className="content-row">
            <div className="recommendations-wrapper">
              <Recommendations 
                customerId={selectedCustomer} 
                godownCode={selectedGodown}
              />
            </div>
          </div>
        </main>
        
        <footer className="footer">
          <div className="container">
            <p>&copy; 2024 Supermarket Recommendation System. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </ProductNameProvider>
  );
}

export default App;
