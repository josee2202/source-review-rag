/* Código SAS FICTICIO para la demo: modelo de PD de hipotecas de Banco Ficticio S.A. */
%let fecha_ini = '01JAN2020'd;
%let fecha_fin = '31DEC2023'd;

/* 1. Selección de la muestra de desarrollo */
data muestra;
    set dwh.operaciones_hipotecarias;
    where fecha_observacion between &fecha_ini. and &fecha_fin.;
run;

/* 2. Marca de default: más de 90 días de impago o improbabilidad de pago */
%macro marcar_default(tabla);
    data &tabla.;
        set &tabla.;
        default_flag = (dias_impago > 90 or flag_improbable_pago = 1);
    run;
%mend marcar_default;
%marcar_default(muestra);

/* 3. Estimación del modelo logístico */
%macro estimar_pd(tabla, salida);
    proc logistic data=&tabla. descending;
        model default_flag = ltv antiguedad ratio_esfuerzo / selection=stepwise;
        output out=&salida. p=pd_estimada;
    run;
%mend estimar_pd;
%estimar_pd(muestra, muestra_pd);

/* 4. Asignación de grados de rating por tramos de PD */
data muestra_pd;
    set muestra_pd;
    if pd_estimada < 0.005 then grado = 1;
    else if pd_estimada < 0.01 then grado = 2;
    else if pd_estimada < 0.03 then grado = 3;
    else if pd_estimada < 0.08 then grado = 4;
    else grado = 5;
run;

/* 5. Backtesting: test binomial por grado al 95% */
%macro test_binomial(tabla);
    proc sql;
        create table backtest as
        select grado,
               count(*) as n_operaciones,
               sum(default_flag) as n_defaults,
               mean(pd_estimada) as pd_media
        from &tabla.
        group by grado;
    quit;
    data backtest;
        set backtest;
        limite_sup = quantile('BINOMIAL', 0.95, pd_media, n_operaciones);
        rechazo = (n_defaults > limite_sup);
    run;
%mend test_binomial;
%test_binomial(muestra_pd);

/* 6. Poder discriminante: AUC */
proc logistic data=muestra_pd descending;
    model default_flag = pd_estimada / nofit;
    roc 'PD estimada' pred=pd_estimada;
run;
