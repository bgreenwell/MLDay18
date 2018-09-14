remark.macros.scale = function (percentage) {
  var url = this;
  return '<img src="' + url + '" style="width: ' + percentage + '" />';
};


remark.macros.color = function (color) {
  var text = this;
  return '<span style="color:' + color + '">"' + text + '"</span>';
};
