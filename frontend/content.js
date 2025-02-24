import { summarizeText } from './summarizer.js';

// Create context menu item
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "summarizeText",
    title: "Summarize with AI",
    contexts: ["selection"]
  });
});

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "summarizeText") {
    const selectedText = info.selectionText;
    
    // Create loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'ai-summarizer-loading';
    loadingDiv.textContent = 'Generating summary...';
    loadingDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 15px;
      background: #f0f0f0;
      border: 1px solid #ccc;
      border-radius: 5px;
      z-index: 10000;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    `;
    document.body.appendChild(loadingDiv);
    
    try {
      const summary = await summarizeText(selectedText);
      
      // Remove loading indicator
      loadingDiv.remove();
      
      // Create summary notification
      const notification = document.createElement('div');
      notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        max-width: 300px;
        padding: 15px;
        background: white;
        border: 1px solid #ccc;
        border-radius: 5px;
        z-index: 10000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
      `;
      
      notification.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold;">Summary:</div>
        <div>${summary}</div>
        <button style="
          position: absolute;
          top: 5px;
          right: 5px;
          border: none;
          background: none;
          cursor: pointer;
          font-size: 16px;
        ">×</button>
      `;
      
      // Add close button functionality
      const closeButton = notification.querySelector('button');
      closeButton.onclick = () => notification.remove();
      
      // Auto-remove after 30 seconds
      setTimeout(() => notification.remove(), 30000);
      
      document.body.appendChild(notification);
    } catch (error) {
      console.error('Summarization failed:', error);
      loadingDiv.remove();
      
      // Show error notification
      const errorNotification = document.createElement('div');
      errorNotification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px;
        background: #ffe6e6;
        border: 1px solid #ff9999;
        border-radius: 5px;
        z-index: 10000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
      `;
      errorNotification.textContent = 'Failed to generate summary. Please try again.';
      
      document.body.appendChild(errorNotification);
      setTimeout(() => errorNotification.remove(), 5000);
    }
  }
});

// Add listener for text selection
document.addEventListener('mouseup', () => {
  const selectedText = window.getSelection().toString().trim();
  // Only show context menu if text is selected
  if (selectedText) {
    chrome.contextMenus.update("summarizeText", {
      visible: true
    });
  } else {
    chrome.contextMenus.update("summarizeText", {
      visible: false
    });
  }
});
