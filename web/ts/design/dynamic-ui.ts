function openTab(event: MouseEvent | null, tabId: string | null): void {
    const tabs = document.querySelectorAll('.tab') as NodeListOf<HTMLElement>;
    const tabContents = document.querySelectorAll('.tab-content') as NodeListOf<HTMLElement>;

    tabs.forEach(tab => {
        tab.classList.remove('active');
    });

    tabContents.forEach(content => {
        content.classList.remove('active');
    });

    if (tabId) {
        const tabContentElement = document.getElementById(tabId + '-content') as HTMLElement | null;
        if (tabContentElement) {
            tabContentElement.classList.add('active');
        }
    }
    if (event && event.currentTarget) {
        const currentTab = event.currentTarget as HTMLElement;
        currentTab.classList.add('active');
    }
}

let tabCount = 0;

function addTab(): void {
    const url = prompt("请输入网页地址:");
    if (url) {
        addTabWithUrl(url, `Tab ${tabCount + 1}`);
    }
}

function addTabWithUrl(url: string, title: string): void {
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
    const tabContainer = document.getElementById('tab-container') as HTMLElement;
    tabContainer.appendChild(newTab);

    const newTabContent = document.createElement('div');
    newTabContent.className = 'tab-design';
    newTabContent.id = 'tab' + tabCount + '-content';

    const newIframe = document.createElement('iframe');
    newIframe.className = 'tab-frame';
    newIframe.src = url;
    newTabContent.appendChild(newIframe);

    const designContainer = document.getElementById('design') as HTMLElement;
    designContainer.appendChild(newTabContent);

    // Switch to the new tab
    openTab(null, 'tab' + tabCount);
    saveTabsToLocalStorage();
}

function closeTab(event: MouseEvent, tabId: string): void {
    event.stopPropagation();
    const confirmClose = confirm("确定要关闭这个标签页吗？");
    if (confirmClose) {
        const tab = document.getElementById(tabId) as HTMLElement;
        const tabContent = document.getElementById(tabId + '-content') as HTMLElement;
        if (tab && tabContent) {
            tab.remove();
            tabContent.remove();
            saveTabsToLocalStorage();
        }
    }
}

function saveTabsToLocalStorage(): void {
    const tabs: { id: string; title: string; url: string }[] = [];
    const tabElements = document.getElementsByClassName('tab') as HTMLCollectionOf<HTMLElement>;

    for (let i = 0; i < tabElements.length; i++) {
        const tabId = tabElements[i].id;
        const title = tabElements[i].textContent?.replace('X', '').trim() || '';
        const tabContent = document.getElementById(tabId + '-content') as HTMLElement | null;
        const iframe = tabContent?.getElementsByTagName('iframe')[0] as HTMLIFrameElement;
        const url = iframe?.src || '';
        tabs.push({ id: tabId, title, url });
    }

    localStorage.setItem('tabs', JSON.stringify(tabs));
}

function restoreTabsFromLocalStorage(): void {
    const savedTabs = localStorage.getItem('tabs');
    if (savedTabs) {
        const tabs = JSON.parse(savedTabs) as { title: string; url: string }[];
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