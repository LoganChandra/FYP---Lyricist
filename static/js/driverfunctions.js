function addDataset(data) {
    adddata = {
        label: data.artistname,
        backgroundColor: 'rgb(255, 99, 132, 0.2)',
        data: data.datasets[0].data
    }
    chart.data.datasets.push({
        data: adddata
    });
    chart.update();
}

function removeDataset() {
    chart.data.datasets.shift();
    chart.update();
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }