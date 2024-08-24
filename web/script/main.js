function showContent(frame_id,menu_class_name,page, element) {
    // Update the iframe's src attribute to load the corresponding page
    document.getElementById(frame_id).src = page;
    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName(menu_class_name);
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}