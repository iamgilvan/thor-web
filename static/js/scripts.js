function yesnoCheck() {
    var element = document.querySelectorAll('[id=matrixPertinence]');
    var elementInput = document.querySelectorAll('[id=matrixPertinenceValue]');
    if (document.getElementById('yesCheck').checked) {

        for(var i = 0; i < element.length; i++){element[i].style.display = 'block'; }
        for(var i = 0; i < elementInput.length; i++){elementInput[i].required = true; }
    }
    else {
        for(var i = 0; i < element.length; i++){element[i].style.display = 'none';}
        for(var i = 0; i < elementInput.length; i++){elementInput[i].required = false;}
    }
}

function changeAssignment(){
    var selectBox = document.getElementById("selectBox");
    var selectedValue = selectBox.options[selectBox.selectedIndex].value;
    if(selectedValue == '1')
    {
        document.getElementById("weight").style.display = 'block';
        document.getElementById("razao").style.display = 'none';
        document.getElementById("escala").style.display = 'none';
    }else if (selectedValue == '2'){
        document.getElementById("weight").style.display = 'none';
        document.getElementById("razao").style.display = 'none';
        document.getElementById("escala").style.display = 'block';
    }
    else if (selectedValue == '3'){
        document.getElementById("weight").style.display = 'none';
        document.getElementById("razao").style.display = 'block';
        document.getElementById("escala").style.display = 'none';
    }
}

function showTca(){
    if (document.getElementById('yesnebulosa').checked) {
        document.getElementById("tcan").style.display = 'block';
    }else{
        document.getElementById("tcan").style.display = 'none';
    }
}

function showW(){
    if (document.getElementById('yescontinue').checked) {
        document.getElementById("w").style.display = 'block';
    }else{
        document.getElementById("w").style.display = 'none';
    }
}
function checkForm(){
    var elementP = document.querySelectorAll('[id=p]');
    var elementQ = document.querySelectorAll('[id=q]');
    for(var i = 0; i < elementP.length; i++)
    {
        valueP = elementP[i].value;
        valueQ = elementQ[i].value;
        if (parseFloat(valueQ) > parseFloat(valueP))
        {
            alert("P values must be greater than Q values");
            return false;
        }
    }
    return true;
}

function goBack() {
    window.history.back();
}