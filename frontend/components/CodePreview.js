import { useState } from 'react';
import styles from '../styles/CodePreview.module.css';

const CodePreview = ({ snippets }) => {
  const [activeSnippet, setActiveSnippet] = useState(0);

  // If there are no snippets, show a placeholder
  if (!snippets || snippets.length === 0) {
    return (
      <div className={styles.codePreview}>
        <div className={styles.emptyState}>
          <h3>Code Preview</h3>
          <p>No code snippets to display yet.</p>
          <p>Ask a question that might involve code examples!</p>
        </div>
      </div>
    );
  }

  // Function to copy code to clipboard
  const copyToClipboard = (code) => {
    navigator.clipboard.writeText(code).then(
      () => {
        // Show a temporary "Copied!" message
        const copyButton = document.getElementById('copy-button');
        if (copyButton) {
          const originalText = copyButton.innerText;
          copyButton.innerText = 'Copied!';
          setTimeout(() => {
            copyButton.innerText = originalText;
          }, 2000);
        }
      },
      (err) => {
        console.error('Could not copy text: ', err);
      }
    );
  };

  return (
    <div className={styles.codePreview}>
      <h3>Code Preview</h3>
      
      {/* Tabs for multiple snippets */}
      {snippets.length > 1 && (
        <div className={styles.tabs}>
          {snippets.map((snippet, index) => (
            <button
              key={index}
              className={`${styles.tab} ${index === activeSnippet ? styles.activeTab : ''}`}
              onClick={() => setActiveSnippet(index)}
            >
              {snippet.language || 'Code'} {index + 1}
            </button>
          ))}
        </div>
      )}
      
      {/* Code display */}
      <div className={styles.codeContainer}>
        <div className={styles.codeHeader}>
          <span className={styles.language}>{snippets[activeSnippet].language || 'code'}</span>
          <button
            id="copy-button"
            className={styles.copyButton}
            onClick={() => copyToClipboard(snippets[activeSnippet].code)}
          >
            Copy
          </button>
        </div>
        <pre className={styles.codeBlock}>
          <code>{snippets[activeSnippet].code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodePreview;