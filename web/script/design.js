
function openTab(event, tabId) {
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.classList.remove('active');
        });

        tabContents.forEach(content => {
            content.classList.remove('active');
        });
        if(tabId) {
            document.getElementById(tabId + '-content').classList.add('active');
        }
        if(event) {
            event.currentTarget.classList.add('active');
        }
    }

let tabCount = 0;
function addTab() {
    const url = prompt("请输入网页地址:");
    if (url) {
        addTabWithUrl(url, `Tab ${tabCount + 1}`);
    }
}

function addTabWithUrl(url, title) {
    tabCount++;
    const newTab = document.createElement('div');
    newTab.className = 'tab';
    newTab.id = 'tab' + tabCount;
    newTab.setAttribute('onclick', `openTab(event, 'tab${tabCount}')`);

    const closeButton = document.createElement('span');
    closeButton.textContent = 'X';
    closeButton.className = 'close-tab';
    closeButton.setAttribute('onclick', `closeTab(event, 'tab${tabCount}')`);

    newTab.textContent = title;
    newTab.appendChild(closeButton);
    document.getElementById('tab-design-n').appendChild(newTab);

    const newTabContent = document.createElement('div');
    newTabContent.className = 'tab-design';
    newTabContent.id = 'tab' + tabCount + '-design';

    const newIframe = document.createElement('iframe');
    newIframe.className = 'tab-frame';
    newIframe.src = url;
    newTabContent.appendChild(newIframe);

    document.getElementById('design').appendChild(newTabContent);
    // Switch to the new tab
    openTab(null, 'tab' + tabCount);
    saveTabsToLocalStorage();
}
function closeTab(event, tabId) {
    event.stopPropagation();
    const confirmClose = confirm("确定要关闭这个标签页吗？");
    if (confirmClose) {
        const tab = document.getElementById(tabId);
        const tabContent = document.getElementById(tabId + '-content');
        tab.remove();
        tabContent.remove();
        saveTabsToLocalStorage();
    }
}

function saveTabsToLocalStorage() {
    const tabs = [];
    const tabElements = document.getElementsByClassName('tab');
    for (let i = 0; i < tabElements.length; i++) {
        const tabId = tabElements[i].id;
        const title = tabElements[i].textContent.replace('X', '').trim();
        const url = document.getElementById(tabId + '-content').getElementsByTagName('iframe')[0].src;
        tabs.push({ id: tabId, title: title, url: url });
    }
    localStorage.setItem('tabs', JSON.stringify(tabs));
}


function restoreTabsFromLocalStorage() {
    const tabs = JSON.parse(localStorage.getItem('tabs'));
    if (tabs) {
        tabs.forEach(tab => {
            const { title, url } = tab;
            addTabWithUrl(url, title);
        });
    }
}

// Call this function when the page loads
window.onload = () => {
    restoreTabsFromLocalStorage();
};